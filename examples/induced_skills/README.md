# induced_skills/

This directory is populated by `python examples/worked_example.py`, which writes:

  - `worked_example_library.json`        the full SkillLibrary in the format used by
                                         `SkillLibrary.load()`.
  - `click_text.json`, `type_in_field.json`, `search_and_filter.json`,
    `checkout_cheapest_in_category.json`   one file per induced skill, useful for
                                            human inspection.

If you run the full WebArena / VWA / OM2W training loop with `scripts/run_training.py`, the same kind of artifacts get written under `runs/<bench>/iterK/library.json` for each iteration K.
