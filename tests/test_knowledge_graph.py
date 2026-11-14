"""
Tests for CandidateKnowledgeGraph (NetworkX-backed entity graph).

Covers:
- Node creation: candidate, skill, employer, title, institution
- Edge creation: HAS_SKILL, WORKED_AT, HELD_TITLE, STUDIED_AT
- Cross-candidate edges: SHARES_SKILLS, COMMON_EMPLOYER
- Subgraph extraction (ego graph)
- Skill querying
- JSON-serializable graph output format
"""
from __future__ import annotations

import pytest
from data_transformer.graph.builder import CandidateKnowledgeGraph
from data_transformer.schema.canonical import (
    CandidateRecord,
    FieldValue,
    EducationEntry,
    ExperienceEntry,
    SkillEntry,
)


def _make_candidate(
    cid: str,
    name: str,
    skills: list[str],
    companies: list[str],
    titles: list[str],
    schools: list[str],
) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=cid,
        full_name=FieldValue(value=name, confidence=1.0),
        skills=[SkillEntry(name=s, normalized=s.lower(), confidence=0.9) for s in skills],
        experience=[
            ExperienceEntry(company=c, title=t, start="2020-01", end="2022-01")
            for c, t in zip(companies, titles)
        ],
        education=[
            EducationEntry(institution=sch, degree="B.S. CS", end_year="2019")
            for sch in schools
        ],
        overall_confidence=0.95,
    )


class TestCandidateKnowledgeGraph:
    def test_single_candidate_nodes_and_edges(self):
        kg = CandidateKnowledgeGraph()
        c = _make_candidate(
            cid="c1",
            name="Alice Smith",
            skills=["Python", "PostgreSQL"],
            companies=["Acme Corp"],
            titles=["Software Engineer"],
            schools=["MIT"],
        )
        kg.add_candidate(c)

        # Candidate node
        cand_id = "candidate:c1"
        assert cand_id in kg.graph
        assert kg.graph.nodes[cand_id]["label"] == "Alice Smith"
        assert kg.graph.nodes[cand_id]["node_type"] == "candidate"

        # Skills nodes and edges
        assert "skill:python" in kg.graph
        assert "skill:postgresql" in kg.graph
        assert kg.graph.has_edge(cand_id, "skill:python")

        # Employer and Title
        assert "employer:acme corp" in kg.graph
        assert "title:software engineer" in kg.graph
        assert kg.graph.has_edge(cand_id, "employer:acme corp")
        assert kg.graph.has_edge(cand_id, "title:software engineer")

        # School
        assert "institution:mit" in kg.graph
        assert kg.graph.has_edge(cand_id, "institution:mit")

    def test_cross_candidate_shared_skills_and_employer(self):
        kg = CandidateKnowledgeGraph()
        c1 = _make_candidate(
            cid="c1",
            name="Alice",
            skills=["Python", "Docker", "AWS", "FastAPI"],
            companies=["Google"],
            titles=["Engineer"],
            schools=["Stanford"],
        )
        c2 = _make_candidate(
            cid="c2",
            name="Bob",
            skills=["Python", "Docker", "AWS", "React"],
            companies=["Google"],
            titles=["Architect"],
            schools=["Berkeley"],
        )

        kg.build_from_candidates([c1, c2])

        # Both share 3 skills: Python, Docker, AWS -> SHARES_SKILLS edge
        # Both share employer Google -> COMMON_EMPLOYER edge
        c1_id = "candidate:c1"
        c2_id = "candidate:c2"
        assert kg.graph.has_edge(c1_id, c2_id)

        edges = kg.graph.get_edge_data(c1_id, c2_id)
        rels = [data.get("relationship") for data in edges.values()]
        assert "SHARES_SKILLS" in rels
        assert "COMMON_EMPLOYER" in rels

    def test_get_candidate_subgraph(self):
        kg = CandidateKnowledgeGraph()
        c = _make_candidate(
            cid="c1",
            name="Alice",
            skills=["Python"],
            companies=["Acme"],
            titles=["Developer"],
            schools=["Harvard"],
        )
        kg.add_candidate(c)
        subgraph = kg.get_candidate_subgraph("c1", depth=1)

        assert "nodes" in subgraph
        assert "links" in subgraph
        node_ids = [n["id"] for n in subgraph["nodes"]]
        assert "candidate:c1" in node_ids
        assert "skill:python" in node_ids
        assert "employer:acme" in node_ids

    def test_get_nonexistent_candidate_subgraph(self):
        kg = CandidateKnowledgeGraph()
        subgraph = kg.get_candidate_subgraph("nonexistent")
        assert subgraph == {"nodes": [], "links": []}

    def test_query_shared_skills(self):
        kg = CandidateKnowledgeGraph()
        c1 = _make_candidate("c1", "Alice", ["Python"], [], [], [])
        c2 = _make_candidate("c2", "Bob", ["Python"], [], [], [])
        c3 = _make_candidate("c3", "Charlie", ["Rust"], [], [], [])
        kg.build_from_candidates([c1, c2, c3])

        python_devs = kg.query_shared_skills("Python")
        cand_ids = {d["candidate_id"] for d in python_devs}
        assert cand_ids == {"c1", "c2"}

        rust_devs = kg.query_shared_skills("Rust")
        assert len(rust_devs) == 1
        assert rust_devs[0]["candidate_id"] == "c3"

        unknown_skill = kg.query_shared_skills("NonexistentSkill")
        assert unknown_skill == []

    def test_get_full_graph(self):
        kg = CandidateKnowledgeGraph()
        c = _make_candidate("c1", "Alice", ["Python"], ["Corp"], ["Dev"], ["School"])
        kg.add_candidate(c)

        full = kg.get_full_graph(max_nodes=10)
        assert len(full["nodes"]) >= 5
        assert len(full["links"]) >= 4
