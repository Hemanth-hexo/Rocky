"""Stub. openwakeword's train.py unconditionally does
`from generate_samples import generate_samples` at the top of main(),
regardless of which step flag is passed — even for --augment_clips/--train_model,
which never call it. This project never uses piper (Kokoro generates the
positive/negative clips instead, see ../../generate_rocky_clips.py), so this
just satisfies the import without installing the real piper-sample-generator."""


def generate_samples(*args, **kwargs):
    raise NotImplementedError(
        "piper-based generation is disabled in this local training setup; "
        "only --augment_clips and --train_model are used, which never call this."
    )
