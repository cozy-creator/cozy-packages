Ideogram 4 is provided under and subject to the Ideogram Non-Commercial Model Agreement available at https://github.com/ideogram-oss/ideogram4/model_licenses/LICENSE-IDEOGRAM-4-NON-COMMERCIAL. All rights reserved. Copyright © Ideogram, Inc.

Cozy's bf16 and int8 lanes of Ideogram 4 are modified versions of its fp8 weights: the bf16 lane decodes them, and the int8 lane re-encodes the blocks' linears as int8 ConvRot. The text encoder, Qwen3-VL-8B-Instruct, is Qwen's, under Apache-2.0.
