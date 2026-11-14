"""
Candidate Knowledge Graph Builder using NetworkX.

Constructs multi-relational graphs linking:
  - Candidates <-> Skills
  - Candidates <-> Employers
  - Candidates <-> Educational Institutions
  - Candidates <-> Job Titles
  - Candidates <-> Certifications
  - Candidates <-> Candidates (Shared skills, overlapping employment)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
import networkx as nx

from data_transformer.schema.canonical import CandidateRecord


class CandidateKnowledgeGraph:
    """
    In-memory multi-relational knowledge graph for candidate intelligence.
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def build_from_candidates(self, candidates: List[CandidateRecord]) -> None:
        """Populates graph from a list of canonical CandidateRecords."""
        self.graph.clear()
        for cand in candidates:
            self.add_candidate(cand)
        self._compute_cross_candidate_edges()

    def add_candidate(self, candidate: CandidateRecord) -> None:
        cand_id = f"candidate:{candidate.candidate_id}"
        name = candidate.full_name.value or "Unknown Candidate"
        confidence = candidate.overall_confidence

        # Add Candidate Node
        self.graph.add_node(
            cand_id,
            id=cand_id,
            label=name,
            node_type="candidate",
            confidence=confidence,
            candidate_id=candidate.candidate_id,
        )

        # 1. Skills
        for skill in candidate.skills:
            skill_name = skill.name.strip()
            if not skill_name:
                continue
            s_id = f"skill:{skill.normalized or skill_name.lower()}"
            self.graph.add_node(s_id, id=s_id, label=skill_name, node_type="skill")
            self.graph.add_edge(
                cand_id,
                s_id,
                relationship="HAS_SKILL",
                confidence=skill.confidence,
            )

        # 2. Employers & Job Titles
        for exp in candidate.experience:
            company = (exp.company or "").strip()
            title = (exp.title or "").strip()
            if company:
                comp_id = f"employer:{company.lower()}"
                self.graph.add_node(comp_id, id=comp_id, label=company, node_type="employer")
                self.graph.add_edge(
                    cand_id,
                    comp_id,
                    relationship="WORKED_AT",
                    title=title,
                    start=exp.start,
                    end=exp.end,
                    is_current=exp.is_current,
                )

            if title:
                title_id = f"title:{title.lower()}"
                self.graph.add_node(title_id, id=title_id, label=title, node_type="title")
                self.graph.add_edge(
                    cand_id,
                    title_id,
                    relationship="HELD_TITLE",
                    company=company,
                )

        # 3. Education
        for edu in candidate.education:
            inst = (edu.institution or "").strip()
            degree = (edu.degree or "").strip()
            if inst:
                inst_id = f"institution:{inst.lower()}"
                self.graph.add_node(inst_id, id=inst_id, label=inst, node_type="institution")
                self.graph.add_edge(
                    cand_id,
                    inst_id,
                    relationship="STUDIED_AT",
                    degree=degree,
                    year=edu.end_year,
                )

    def _compute_cross_candidate_edges(self) -> None:
        """Finds peer-to-peer relationships between candidates."""
        candidate_nodes = [
            n for n, data in self.graph.nodes(data=True) if data.get("node_type") == "candidate"
        ]

        for i in range(len(candidate_nodes)):
            c1 = candidate_nodes[i]
            skills1 = {
                target for _, target, data in self.graph.out_edges(c1, data=True)
                if data.get("relationship") == "HAS_SKILL"
            }
            companies1 = {
                target for _, target, data in self.graph.out_edges(c1, data=True)
                if data.get("relationship") == "WORKED_AT"
            }

            for j in range(i + 1, len(candidate_nodes)):
                c2 = candidate_nodes[j]
                skills2 = {
                    target for _, target, data in self.graph.out_edges(c2, data=True)
                    if data.get("relationship") == "HAS_SKILL"
                }
                companies2 = {
                    target for _, target, data in self.graph.out_edges(c2, data=True)
                    if data.get("relationship") == "WORKED_AT"
                }

                # Shared skills
                shared_s = skills1 & skills2
                if len(shared_s) >= 3:
                    self.graph.add_edge(
                        c1,
                        c2,
                        relationship="SHARES_SKILLS",
                        weight=len(shared_s),
                        count=len(shared_s),
                        skills=[s.replace("skill:", "") for s in shared_s],
                    )

                # Shared employer (potential alumni/colleagues)
                shared_c = companies1 & companies2
                if shared_c:
                    self.graph.add_edge(
                        c1,
                        c2,
                        relationship="COMMON_EMPLOYER",
                        employers=[c.replace("employer:", "").title() for c in shared_c],
                    )

    def get_candidate_subgraph(self, candidate_id: str, depth: int = 1) -> Dict[str, Any]:
        """Returns JSON-serializable node/link format for a candidate's neighborhood."""
        cand_id = f"candidate:{candidate_id}"
        if cand_id not in self.graph:
            return {"nodes": [], "links": []}

        # Ego graph (neighborhood)
        sub = nx.ego_graph(self.graph, cand_id, radius=depth, undirected=True)
        return self._graph_to_dict(sub)

    def get_full_graph(self, max_nodes: int = 250) -> Dict[str, Any]:
        """Returns JSON-serializable representation of the graph up to max_nodes."""
        nodes = list(self.graph.nodes())[:max_nodes]
        sub = self.graph.subgraph(nodes)
        return self._graph_to_dict(sub)

    def query_shared_skills(self, skill_name: str) -> List[Dict[str, Any]]:
        """Finds all candidates possessing a specific skill."""
        s_id = f"skill:{skill_name.lower()}"
        if s_id not in self.graph:
            return []

        results = []
        for cand_id, _, data in self.graph.in_edges(s_id, data=True):
            cand_data = self.graph.nodes[cand_id]
            results.append({
                "candidate_id": cand_data.get("candidate_id"),
                "name": cand_data.get("label"),
                "confidence": data.get("confidence", 0.0),
            })
        return results

    def _graph_to_dict(self, g: nx.Graph) -> Dict[str, Any]:
        nodes = []
        for n, data in g.nodes(data=True):
            nodes.append({
                "id": str(n),
                "label": data.get("label", str(n)),
                "type": data.get("node_type", "unknown"),
                "confidence": data.get("confidence"),
                "candidate_id": data.get("candidate_id"),
            })

        links = []
        for u, v, data in g.edges(data=True):
            links.append({
                "source": str(u),
                "target": str(v),
                "relationship": data.get("relationship", "CONNECTED"),
                "title": data.get("title"),
                "weight": data.get("weight", 1.0),
            })

        return {"nodes": nodes, "links": links}
