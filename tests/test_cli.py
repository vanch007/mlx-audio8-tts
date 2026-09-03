from mlx_audio8_tts.cli import DEFAULT_MODEL, create_parser


def test_published_model_is_the_cli_default():
    parser = create_parser()

    for command in ("generate", "audit", "serve"):
        argv = [command]
        if command == "generate":
            argv += ["--text", "hello"]
        args = parser.parse_args(argv)
        assert args.model == DEFAULT_MODEL

    assert DEFAULT_MODEL == "vanch007/Audio8-TTS-MLX-8bit"
    assert "/Users/" not in DEFAULT_MODEL
