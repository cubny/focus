# Gemini Instructions

## Documentation Updates

> [!IMPORTANT]
> **ALWAYS** update `README.md` and the `--help` output (by modifying the CLI code) immediately when adding new user-facing features. Do not wait for a separate task.

When implementing new CLI features:
1. Add the feature to the CLI code (using `click`).
2. Verify `--help` shows the new options.
3. Update `README.md` to include description and usage examples.
