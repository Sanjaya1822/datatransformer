"""
Candidate Clustering module.
Groups RawRecords into distinct CandidateClusters based on identity scores.

Clustering strategy:
- Pass 1: Deterministic blocking on email, phone, GitHub, LinkedIn only
  (Name is NOT used as a blocking key to avoid false merges)
- Pass 2: Fuzzy matching between remaining distinct clusters
  Auto-merge only at score >= 90.0 (requires hard identifier match)
  Flag for review at score >= 70.0 (soft signals)
"""
from typing import List, Dict, Any, Set
from dataclasses import dataclass, field
import uuid

from data_transformer.schema.canonical import RawRecord
from data_transformer.pipeline.identity_resolver import compute_identity_score


@dataclass
class CandidateCluster:
    cluster_id: str
    records: List[RawRecord] = field(default_factory=list)
    requires_review: bool = False
    review_reason: str = ""
    scores: Dict[str, float] = field(default_factory=dict)  # Pairwise scores within cluster


class CandidateClusterer:
    """
    Groups raw records into candidate clusters.

    Thresholds:
      auto_merge_threshold: score >= this → definitive same person (requires hard ID match)
      review_threshold: score >= this → uncertain, flag for human review
    """
    def __init__(
        self,
        auto_merge_threshold: float = 90.0,
        manual_review_threshold: float = 70.0,
        review_threshold: Optional[float] = None,
    ):
        self.auto_merge_threshold = auto_merge_threshold
        self.manual_review_threshold = review_threshold if review_threshold is not None else manual_review_threshold
        self.review_threshold = self.manual_review_threshold

    def cluster(self, records: List[RawRecord]) -> List[CandidateCluster]:
        """
        Takes a list of records and clusters them.
        Uses connected components for clustering.
        """
        if not records:
            return []

        if len(records) == 1:
            return [CandidateCluster(cluster_id=uuid.uuid4().hex[:8], records=list(records))]

        # ── Pass 1: Deterministic Blocking on hard identifiers only ──────────
        # NOTE: Full name is intentionally excluded from blocking keys.
        #       Two different people can share the same name.
        cluster_map: Dict[str, str] = {}
        record_clusters: Dict[str, CandidateCluster] = {}

        def merge_clusters(id1: str, id2: str) -> str:
            if id1 == id2:
                return id1
            c1 = record_clusters[id1]
            c2 = record_clusters[id2]
            c1.records.extend(c2.records)
            if c2.requires_review:
                c1.requires_review = True
                c1.review_reason += " | " + c2.review_reason
            # Redirect all pointers
            for k, v in list(cluster_map.items()):
                if v == id2:
                    cluster_map[k] = id1
            del record_clusters[id2]
            return id1

        for rec in records:
            fields = rec.fields
            emails = fields.get("emails", [])
            phones = fields.get("phones", [])
            links = fields.get("links", [])

            # Only hard-identifier blocking keys (NOT name)
            keys_to_check = []
            for e in emails:
                if e:
                    keys_to_check.append(f"email:{str(e).strip().lower()}")
            for p in phones:
                if p:
                    keys_to_check.append(f"phone:{str(p).strip()}")
            for link in links:
                if isinstance(link, dict):
                    url = link.get("url", "")
                    link_type = link.get("type", "")
                    if url and link_type in ("github", "linkedin"):
                        keys_to_check.append(f"{link_type}:{url.lower()}")

            matched_cluster_ids = set()
            for k in keys_to_check:
                if k in cluster_map:
                    matched_cluster_ids.add(cluster_map[k])

            target_cid = None
            if not matched_cluster_ids:
                # New isolated cluster — no hard ID overlap
                target_cid = uuid.uuid4().hex[:8]
                record_clusters[target_cid] = CandidateCluster(
                    cluster_id=target_cid, records=[rec]
                )
            else:
                matched_list = list(matched_cluster_ids)
                target_cid = matched_list[0]
                record_clusters[target_cid].records.append(rec)
                for other_cid in matched_list[1:]:
                    if other_cid in record_clusters:
                        target_cid = merge_clusters(target_cid, other_cid)

            # Register all hard-ID keys for this cluster
            for k in keys_to_check:
                cluster_map[k] = target_cid

            # If no hard identifiers, this record gets its own isolated cluster
            if not keys_to_check:
                if target_cid is None:
                    target_cid = uuid.uuid4().hex[:8]
                    record_clusters[target_cid] = CandidateCluster(
                        cluster_id=target_cid, records=[rec]
                    )

        # ── Pass 2: Cross-cluster fuzzy comparison ────────────────────────────
        clusters_list = list(record_clusters.values())
        merged_in_pass2: Set[str] = set()
        final_clusters: List[CandidateCluster] = []

        for i in range(len(clusters_list)):
            ci = clusters_list[i]
            if ci.cluster_id in merged_in_pass2:
                continue

            for j in range(i + 1, len(clusters_list)):
                cj = clusters_list[j]
                if cj.cluster_id in merged_in_pass2:
                    continue

                # Compare records across clusters to find best identity match
                best_score = 0.0
                for r1 in ci.records:
                    for r2 in cj.records:
                        s = compute_identity_score(r1, r2)
                        if s > best_score:
                            best_score = s

                if best_score >= self.auto_merge_threshold:
                    # Definitive match — auto-merge
                    ci.records.extend(cj.records)
                    merged_in_pass2.add(cj.cluster_id)
                elif best_score >= self.review_threshold:
                    # Uncertain — flag for human review but do NOT auto-merge
                    ci.requires_review = True
                    ci.review_reason = (
                        f"Uncertain match with cluster {cj.cluster_id} "
                        f"(score={best_score:.1f}): manual review required."
                    )

            final_clusters.append(ci)

        return final_clusters
