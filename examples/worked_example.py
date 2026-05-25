"""App. B: a worked three-level recursive skill.

This script reproduces the worked example from Appendix B of the paper. It
constructs a depth-3 skill `checkout_cheapest_in_category` that calls a depth-2
skill `search_and_filter` and two depth-1 skills `click_text` and
`type_in_field`. The constructed library is then saved to JSON.

Run:
    python examples/worked_example.py
"""

from __future__ import annotations

import json
import os

from scaffold.composition.recursive import compute_depth, depth_distribution
from scaffold.core.library import SkillLibrary
from scaffold.core.skill import Parameter, ParameterType, Skill
from scaffold.utils.embedding import HashEmbedder


def build_library() -> SkillLibrary:
    lib = SkillLibrary(embedder=HashEmbedder())

    # ----- Level 1 (depth 1): atomic skills -----------------------------
    click_text = Skill(
        name="click_text",
        description="Click an element with the given visible text.",
        parameters=[Parameter("text", ParameterType.STRING)],
        precondition="any(getattr(e, 'text', '') == text for e in (obs.dom or []))",
        body=(
            "el = find(obs.dom, text=text)\n"
            "primitive_click(el)"
        ),
        postcondition="True",   # in real runs: page-change or pressed-state check
        depth=1,
        iteration_introduced=1,
    )
    lib.add(click_text)

    type_in_field = Skill(
        name="type_in_field",
        description="Type a value into the input field labeled `label`.",
        parameters=[
            Parameter("label", ParameterType.STRING),
            Parameter("value", ParameterType.STRING),
        ],
        precondition="True",   # real: exists_input(obs.dom, label)
        body=(
            "f = find_input(obs.dom, label=label)\n"
            "primitive_click(f)\n"
            "primitive_type(f, value)"
        ),
        postcondition="True",
        depth=1,
        iteration_introduced=1,
    )
    lib.add(type_in_field)

    # ----- Level 2 (depth 2): mid-level composite -----------------------
    search_and_filter = Skill(
        name="search_and_filter",
        description="Search for a query in a category and apply a sort order.",
        parameters=[
            Parameter("query", ParameterType.STRING),
            Parameter("category", ParameterType.STRING),
            Parameter("sort", ParameterType.STRING, default="price_asc",
                       choices=("price_asc", "price_desc", "rating_desc")),
        ],
        precondition="'shop' in obs.url",
        body=(
            "type_in_field(obs, label='Search', value=query)\n"
            "click_text(obs, text='Search')\n"
            "click_text(obs, text=category)\n"
            "click_text(obs, text=f'Sort: {sort}')"
        ),
        postcondition="True",
        depth=2,   # set explicitly; compute_depth verifies
        iteration_introduced=2,
    )
    # depth check before adding (cycle check happens inside library.add via composition)
    assert compute_depth(search_and_filter, lib) == 2, "depth must be 2"
    lib.add(search_and_filter)

    # ----- Level 3 (depth 3): composite of composite -------------------
    checkout = Skill(
        name="checkout_cheapest_in_category",
        description="Buy the cheapest item in a category, then complete payment.",
        parameters=[
            Parameter("query", ParameterType.STRING),
            Parameter("category", ParameterType.STRING),
            Parameter("payment_token", ParameterType.STRING),
        ],
        precondition="True",   # real: logged_in(obs)
        body=(
            "search_and_filter(obs, query=query, category=category, sort='price_asc')\n"
            "click_text(obs, text='first_result')\n"
            "click_text(obs, text='Add to Cart')\n"
            "click_text(obs, text='Checkout')\n"
            "type_in_field(obs, label='PaymentToken', value=payment_token)\n"
            "click_text(obs, text='Place Order')"
        ),
        postcondition="True",   # real: order_confirmed(obs) AND category match
        depth=3,
        iteration_introduced=3,
    )
    assert compute_depth(checkout, lib) == 3, "depth must be 3"
    lib.add(checkout)
    return lib


def main() -> None:
    lib = build_library()
    print(f"Built library with {lib.total_skills()} skills.")
    print(f"  Mean depth : {lib.mean_depth():.2f}")
    print(f"  Max depth  : {lib.max_depth()}")
    print(f"  Depth dist : {depth_distribution(lib)}")
    print()
    for s in lib:
        print(f"  - {s.name}  (d={s.depth}, params={len(s.parameters)})")

    out_dir = os.path.join(os.path.dirname(__file__), "induced_skills")
    os.makedirs(out_dir, exist_ok=True)
    lib_path = os.path.join(out_dir, "worked_example_library.json")
    lib.save(lib_path)
    print(f"\nSaved library to {lib_path}")

    # Also dump each skill individually for inspection.
    for s in lib:
        with open(os.path.join(out_dir, f"{s.name}.json"), "w") as f:
            json.dump(s.to_dict(), f, indent=2)
    print(f"Saved per-skill JSON to {out_dir}/")


if __name__ == "__main__":
    main()
