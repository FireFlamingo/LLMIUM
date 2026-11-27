# Third-party source

## Fonts

The interface and PDFs bundle unmodified Manrope and IBM Plex Mono from the Google Fonts repository. Both use the SIL Open Font License 1.1. The font files and complete licenses are in `web/fonts/`. The interface loads fonts locally.

## Candle

`src/traced_qwen.rs` derives its Qwen2 forward operations from Hugging Face Candle 0.11.1, `candle-transformers/src/models/qwen2.rs`, at commit `59f8e903ca80a32ea7f8514b39748fc979e69e82`. LLMIUM adds read-only intermediate activation observations and uses the original crate's configuration, linear and normalization primitives. The upstream source is available under MIT OR Apache-2.0; this derived file uses the MIT option. Its permission notice is preserved in [third_party/CANDLE-LICENSE-MIT](third_party/CANDLE-LICENSE-MIT).

[Upstream source](https://github.com/huggingface/candle/blob/59f8e903ca80a32ea7f8514b39748fc979e69e82/candle-transformers/src/models/qwen2.rs) · [Upstream MIT license](https://github.com/huggingface/candle/blob/59f8e903ca80a32ea7f8514b39748fc979e69e82/LICENSE-MIT)

Other Cargo dependencies retain their own licenses and are recorded in `Cargo.lock`.
