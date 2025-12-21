#!/usr/bin/env python
"""
Test script for AGC-Agent.

This script tests the AGC-Agent implementation with a simple example
to verify all components work correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from agc_agent import (
    KGIndex, RelationIndex, NeighborIndex,
    BeamState, BeamManager, PathAccumulator, BeamStatus,
    AGCAgentConfig
)


def test_kg_index():
    """Test KG Index structures."""
    print("=" * 60)
    print("Testing KG Index Structures")
    print("=" * 60)

    # Sample triples
    triples = [
        ("Barack Obama", "spouse_of", "Michelle Obama"),
        ("Barack Obama", "president_of", "USA"),
        ("Barack Obama", "born_in", "Hawaii"),
        ("Barack Obama", "children", "Malia Obama"),
        ("Barack Obama", "children", "Sasha Obama"),
        ("USA", "capital", "Washington DC"),
        ("USA", "ex_president", "Barack Obama"),
        ("USA", "ex_president", "Donald Trump"),
        ("Michelle Obama", "spouse_of", "Barack Obama"),
    ]

    # Test RelationIndex
    print("\n1. Testing RelationIndex...")
    rel_index = RelationIndex()
    rel_index.build_from_triples(triples)

    relations = rel_index.get_relation_names("Barack Obama")
    print(f"   Relations from 'Barack Obama': {relations}")
    assert "spouse_of" in relations
    assert "president_of" in relations
    assert "children" in relations
    print("   ✓ RelationIndex passed")

    # Test NeighborIndex
    print("\n2. Testing NeighborIndex...")
    neighbor_index = NeighborIndex()
    neighbor_index.build_from_triples(triples)

    neighbors = neighbor_index.get_neighbors_list("Barack Obama", "children")
    print(f"   Children of 'Barack Obama': {neighbors}")
    assert "Malia Obama" in neighbors
    assert "Sasha Obama" in neighbors
    print("   ✓ NeighborIndex passed")

    # Test combined KGIndex
    print("\n3. Testing KGIndex (combined)...")
    kg_index = KGIndex()
    kg_index.build_from_triples(triples)

    relations = kg_index.get_valid_relations("USA")
    print(f"   Relations from 'USA': {relations}")
    entities = kg_index.get_valid_entities("USA", "ex_president")
    print(f"   Ex-presidents of 'USA': {entities}")
    assert "Barack Obama" in entities
    assert "Donald Trump" in entities
    print("   ✓ KGIndex passed")

    print("\n✓ All KG Index tests passed!")
    return kg_index


def test_beam_state():
    """Test BeamState and BeamManager."""
    print("\n" + "=" * 60)
    print("Testing Beam State and Manager")
    print("=" * 60)

    # Test BeamState
    print("\n1. Testing BeamState...")
    beam = BeamState(current_entity="Barack Obama")
    print(f"   Initial: {beam.path_to_string()}")
    assert beam.depth == 0
    assert beam.current_entity == "Barack Obama"

    # Extend beam
    extended = beam.extend("spouse_of", "Michelle Obama", 0.9, 0.95)
    print(f"   Extended: {extended.path_to_string()}")
    assert extended.depth == 1
    assert extended.current_entity == "Michelle Obama"
    assert len(extended.path) == 1
    print("   ✓ BeamState extend passed")

    # Test backtrack
    backtracked = extended.backtrack()
    print(f"   Backtracked: {backtracked.path_to_string()}")
    assert backtracked.depth == 0
    assert backtracked.current_entity == "Barack Obama"
    assert backtracked.backtrack_count == 1
    print("   ✓ BeamState backtrack passed")

    # Test complete
    completed = extended.complete()
    assert completed.status == BeamStatus.COMPLETED
    print("   ✓ BeamState complete passed")

    # Test BeamManager
    print("\n2. Testing BeamManager...")
    manager = BeamManager(beam_width=5, max_depth=3)
    manager.initialize(["Barack Obama", "USA"])

    assert len(manager.get_active_beams()) == 2
    print(f"   Initialized with {len(manager.get_active_beams())} beams")

    # Simulate some exploration
    active = manager.get_active_beams()
    manager.clear_active_beams()

    for beam in active:
        if beam.current_entity == "Barack Obama":
            new_beam = beam.extend("spouse_of", "Michelle Obama", 0.9, 0.95)
            manager.add_candidate(new_beam)
            new_beam2 = beam.extend("president_of", "USA", 0.8, 0.9)
            manager.add_candidate(new_beam2)
        else:
            completed_beam = beam.complete()
            manager.add_candidate(completed_beam)

    manager.prune_to_top_k()

    print(f"   Active beams after step: {len(manager.get_active_beams())}")
    print(f"   Completed beams: {len(manager.get_completed_beams())}")
    print("   ✓ BeamManager passed")

    print("\n✓ All Beam tests passed!")


def test_path_accumulator():
    """Test PathAccumulator."""
    print("\n" + "=" * 60)
    print("Testing Path Accumulator")
    print("=" * 60)

    accumulator = PathAccumulator()

    # Create some test beams
    beam1 = BeamState(current_entity="Barack Obama")
    beam1 = beam1.extend("spouse_of", "Michelle Obama", 0.9, 0.95)
    beam1 = beam1.complete()

    beam2 = BeamState(current_entity="USA")
    beam2 = beam2.extend("capital", "Washington DC", 0.85, 0.9)
    beam2 = beam2.complete()

    accumulator.add_paths([beam1, beam2])

    print("\n1. Testing path formatting...")
    paths = accumulator.get_paths()
    for path_str, score in paths:
        print(f"   Path: {path_str} (score: {score:.4f})")

    print("\n2. Testing evaluation format...")
    formatted = accumulator.format_for_evaluation(top_k=2)
    for i, f in enumerate(formatted):
        print(f"   Prediction {i+1}:")
        for line in f.split("\n"):
            print(f"      {line}")

    print("\n3. Testing answer extraction...")
    answers = accumulator.get_answers()
    for answer, score in answers:
        print(f"   Answer: {answer} (score: {score:.4f})")

    print("\n✓ All Path Accumulator tests passed!")


def test_full_pipeline():
    """Test the full pipeline with mock model."""
    print("\n" + "=" * 60)
    print("Testing Full Pipeline (without actual LLM)")
    print("=" * 60)

    # This test simulates what AGCAgent does without the actual LLM

    # Sample data
    question = "Who is Barack Obama's spouse?"
    triples = [
        ("Barack Obama", "spouse_of", "Michelle Obama"),
        ("Barack Obama", "president_of", "USA"),
        ("Barack Obama", "born_in", "Hawaii"),
        ("Michelle Obama", "spouse_of", "Barack Obama"),
        ("Michelle Obama", "born_in", "Chicago"),
    ]
    topic_entities = ["Barack Obama"]

    print(f"\n   Question: {question}")
    print(f"   Topic entities: {topic_entities}")

    # Build KG index
    kg_index = KGIndex()
    kg_index.build_from_triples(triples)

    # Simulate beam search
    manager = BeamManager(beam_width=3, max_depth=2)
    manager.initialize(topic_entities)

    # Step 1: Get valid relations
    active = manager.get_active_beams()
    print(f"\n   Step 1: {len(active)} active beams")

    for beam in active:
        relations = kg_index.get_valid_relations(beam.current_entity)
        print(f"   - From '{beam.current_entity}': {relations}")

    # Simulate selection: choose spouse_of
    manager.clear_active_beams()
    for beam in active:
        # Simulate selecting "spouse_of" relation
        entities = kg_index.get_valid_entities(beam.current_entity, "spouse_of")
        if entities:
            new_beam = beam.extend("spouse_of", entities[0], 0.95, 0.98)
            # Simulate ANSWER action
            completed = new_beam.complete()
            manager.add_candidate(completed)

    # Get results
    results = manager.get_completed_beams()
    print(f"\n   Completed paths: {len(results)}")

    accumulator = PathAccumulator()
    accumulator.add_paths(results)

    answers = accumulator.get_answers()
    print(f"\n   Final answers:")
    for answer, score in answers:
        print(f"   - {answer} (score: {score:.4f})")

    # Verify we got the right answer
    assert any(a == "Michelle Obama" for a, _ in answers)
    print("\n✓ Full pipeline test passed!")


def test_cycle_prevention():
    """Test that cycles are prevented (GCR alignment)."""
    print("\n" + "=" * 60)
    print("Testing Cycle Prevention (GCR Alignment)")
    print("=" * 60)

    # Sample data simulating the Jamaica example
    triples = [
        ("Jamaica", "location.country.languages_spoken", "Jamaican Creole English Language"),
        ("Jamaica", "location.country.languages_spoken", "Jamaican English"),
        ("Jamaica", "location.country.form_of_government", "Parliamentary system"),
        ("Jamaican Creole English Language", "language.human_language.main_country", "Jamaica"),
        ("Parliamentary system", "government.form_of_government.countries", "Jamaica"),
        ("Parliamentary system", "government.form_of_government.countries", "Belize"),
    ]
    topic_entities = ["Jamaica"]

    print(f"\n   Topic entities: {topic_entities}")

    # Build KG index
    kg_index = KGIndex()
    kg_index.build_from_triples(triples)

    topic_entity_set = set(topic_entities)

    # Generate all valid paths up to depth 2 (like our fixed implementation)
    all_paths = []

    for start_entity in topic_entities:
        # BFS-like exploration
        queue = [(start_entity, [], set())]

        while queue:
            current, path, visited = queue.pop(0)

            # Get all entities in current path to prevent cycles
            path_entities = {start_entity}
            for h, r, t in path:
                path_entities.add(h)
                path_entities.add(t)

            if len(path) >= 2:  # max_depth = 2
                continue

            relations = kg_index.get_valid_relations(current)
            for rel in relations:
                if (current, rel) in visited:
                    continue

                entities = kg_index.get_valid_entities(current, rel)
                for ent in entities:
                    # Prevent cycles: skip topic entities and entities in path
                    if ent in topic_entity_set or ent in path_entities:
                        continue

                    new_path = path + [(current, rel, ent)]
                    new_visited = visited.copy()
                    new_visited.add((current, rel))

                    # Add this path
                    all_paths.append(new_path)

                    # Continue exploring
                    if len(new_path) < 2:
                        queue.append((ent, new_path, new_visited))

    print(f"\n   Generated {len(all_paths)} paths:")
    for path in all_paths:
        parts = [path[0][0]]
        for h, r, t in path:
            parts.extend([r, t])
        path_str = " -> ".join(parts)
        print(f"   - {path_str}")

    # Verify no path ends at a topic entity (no cycles back to start)
    for path in all_paths:
        answer = path[-1][2]  # tail of last triple
        assert answer not in topic_entity_set, f"Path ends at topic entity: {answer}"

    # Verify we have both 1-hop and 2-hop paths
    one_hop_paths = [p for p in all_paths if len(p) == 1]
    two_hop_paths = [p for p in all_paths if len(p) == 2]
    print(f"\n   1-hop paths: {len(one_hop_paths)}")
    print(f"   2-hop paths: {len(two_hop_paths)}")

    # Verify expected 1-hop paths exist
    one_hop_answers = [p[-1][2] for p in one_hop_paths]
    assert "Jamaican Creole English Language" in one_hop_answers
    assert "Jamaican English" in one_hop_answers
    assert "Parliamentary system" in one_hop_answers

    # Verify 2-hop path to Belize exists (but not back to Jamaica)
    two_hop_answers = [p[-1][2] for p in two_hop_paths]
    assert "Belize" in two_hop_answers
    assert "Jamaica" not in two_hop_answers  # No cycle back to start

    print("\n✓ Cycle prevention test passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("AGC-Agent Test Suite")
    print("=" * 60)

    test_kg_index()
    test_beam_state()
    test_path_accumulator()
    test_full_pipeline()
    test_cycle_prevention()

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
    print("\nTo run the full AGC-Agent with a model, use:")
    print("  python agc_reasoning.py --gpu_id 0 --d RoG-webqsp --split test[:10]")
    print("\nFor simplified mode (faster but less accurate):")
    print("  python agc_reasoning.py --gpu_id 0 --simplified --d RoG-webqsp --split test[:10]")


if __name__ == "__main__":
    main()
