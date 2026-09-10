/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rolling.cuh"

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf::detail {

// Applies a variable-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& preceding_window,
                                       column_view const& following_window,
                                       size_type min_periods,
                                       rolling_aggregation const& agg,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  if (preceding_window.is_empty() || following_window.is_empty() || input.is_empty()) {
    return cudf::detail::empty_output_for_rolling_aggregation(input, agg);
  }

  CUDF_EXPECTS(preceding_window.type().id() == type_id::INT32 &&
                 following_window.type().id() == type_id::INT32,
               "preceding_window/following_window must have type_id::INT32 type");

  CUDF_EXPECTS(preceding_window.size() == input.size() && following_window.size() == input.size(),
               "preceding_window/following_window size must match input size");

  auto defaults_col =
    cudf::is_dictionary(input.type()) ? dictionary_column_view(input).indices() : input;
  return cudf::detail::rolling_window(input,
                                      empty_like(defaults_col)->view(),
                                      preceding_window.begin<size_type>(),
                                      following_window.begin<size_type>(),
                                      min_periods,
                                      agg,
                                      stream,
                                      mr);
}

std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& default_outputs,
                                       column_view const& window_starts,
                                       column_view const& window_ends,
                                       size_type min_periods,
                                       rolling_aggregation const& agg,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(
    window_starts.type().id() == type_id::INT32 && window_ends.type().id() == type_id::INT32,
    "window_starts/window_ends must have type_id::INT32 type");

  CUDF_EXPECTS(window_starts.size() == window_ends.size(),
               "window_starts/window_ends must have the same size");

  CUDF_EXPECTS(!window_starts.has_nulls() && !window_ends.has_nulls(),
               "window_starts/window_ends must not contain nulls");

  CUDF_EXPECTS(default_outputs.is_empty(),
               "Absolute-bounds rolling does not support default outputs");

  CUDF_EXPECTS(!input.is_empty() || window_starts.is_empty(),
               "Non-empty absolute bounds require a non-empty input column");

  CUDF_EXPECTS(is_supported_absolute_bounds_aggregation(agg.kind),
               "Unsupported absolute-bounds rolling aggregation");

  return cudf::detail::rolling_window(input,
                                      default_outputs,
                                      window_starts.size(),
                                      window_starts.begin<size_type>(),
                                      window_ends.begin<size_type>(),
                                      min_periods,
                                      agg,
                                      stream,
                                      mr);
}

}  // namespace cudf::detail
