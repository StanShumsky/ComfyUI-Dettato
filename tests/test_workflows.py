"""The example workflows must stay loadable: links and widget counts match the node schemas."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("workflow", ["audio2midi"])
def test_example_workflow(nodes, workflow):
    data = json.loads((ROOT / "example_workflows" / f"{workflow}.json").read_text())
    by_id = {n["id"]: n for n in data["nodes"]}
    links = {link[0]: link for link in data["links"]}
    for node in data["nodes"]:
        for slot, inp in enumerate(node.get("inputs") or []):
            if inp.get("link") is None:
                continue
            link = links[inp["link"]]
            assert link[3] == node["id"] and link[4] == slot
            source = by_id[link[1]]["outputs"][link[2]]
            assert source["type"] == link[5] == inp["type"]
            assert inp["link"] in (source["links"] or [])
        for slot, out in enumerate(node.get("outputs") or []):
            for link_id in out.get("links") or []:
                link = links[link_id]
                assert link[1] == node["id"] and link[2] == slot
        cls = nodes.NODE_CLASS_MAPPINGS.get(node["type"])
        if cls is None:
            continue
        required = cls.INPUT_TYPES()["required"]
        linked = {inp["name"] for inp in node.get("inputs") or []}
        widgets = [name for name in required if name not in linked]
        assert len(node["widgets_values"]) == len(widgets), node["type"]
        for name, value in zip(widgets, node["widgets_values"]):
            spec = required[name]
            if isinstance(spec[0], list):
                if len(spec) == 2 and spec[1].get("multi_select"):
                    assert isinstance(value, list) and value and all(v in spec[0] for v in value), (node["type"], name)
                else:
                    assert value in spec[0], (node["type"], name)