#pragma once

#if defined(QWEN35X_VARIANT_4B)
#include "variant_4b.cuh"
#elif defined(QWEN35X_VARIANT_0P8B) || !defined(QWEN35X_VARIANT_4B)
#include "variant_0p8b.cuh"
#endif
