/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <thrust/iterator/counting_iterator.h>

auto all_valid = [](cudf::size_type row) { return true; };
auto even_valid = [](cudf::size_type row) { return (row % 2 == 0); };

template <typename T>
class CopyRangeTypedTestFixture : public cudf::test::BaseFixture {
public:
  static constexpr cudf::size_type column_size{1000};

  void test(cudf::column_view const& source,
            cudf::column_view const& expected,
            cudf::mutable_column_view& target,
            cudf::size_type source_begin, cudf::size_type source_end,
            cudf::size_type target_begin) {
    static_assert(cudf::is_fixed_width<T>(),
                  "this code assumes fixed-width types.");

    // test the out-of-place version first

    const cudf::column_view immutable_view{target};
    auto p_ret =
      cudf::experimental::copy_range(
        source, immutable_view, source_begin, source_end, target_begin);
    cudf::test::expect_columns_equal(*p_ret, expected);

    // test the in-place version second

    EXPECT_NO_THROW(cudf::experimental::copy_range(source, target,
                      source_begin, source_end, target_begin));
    cudf::test::expect_columns_equal(target, expected);
  }
};

TYPED_TEST_CASE(CopyRangeTypedTestFixture, cudf::test::FixedWidthTypes);

TYPED_TEST(CopyRangeTypedTestFixture, CopyWithNulls)
{
  using T = TypeParam;

  cudf::size_type size{CopyRangeTypedTestFixture<T>::column_size};
  cudf::size_type source_begin{9};
  cudf::size_type source_end{size - 50};
  cudf::size_type target_begin{30};
  auto target_end = target_begin + (source_end - source_begin);
  auto row_diff = source_begin - target_begin;

  auto target = cudf::test::fixed_width_column_wrapper<T>(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + size,
    cudf::test::make_counting_transform_iterator(0, all_valid));

  auto source_elements =
    cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return i * 2; });
  auto source = cudf::test::fixed_width_column_wrapper<T>(
    source_elements,
    source_elements + size,
    cudf::test::make_counting_transform_iterator(
      0, even_valid));

  auto expected_elements =
    cudf::test::make_counting_transform_iterator(
      0,
      [target_begin, target_end, row_diff](auto i) {
        return ((i >= target_begin) && (i < target_end)) ?
          (i + row_diff) * 2 : i;
     });
  auto expected = cudf::test::fixed_width_column_wrapper<T>(
    expected_elements,
    expected_elements + size,
    cudf::test::make_counting_transform_iterator(
      0,
      [target_begin, target_end, row_diff](auto i) {
        return ((i >= target_begin) && (i < target_end)) ?
          even_valid(i + row_diff) : all_valid(i);
      }));

  cudf::mutable_column_view target_view{target};
  this->test(source, expected, target_view,
             source_begin, source_end, target_begin);
}

TYPED_TEST(CopyRangeTypedTestFixture, CopyNoNulls)
{
  using T = TypeParam;

  cudf::size_type size{CopyRangeTypedTestFixture<T>::column_size};
  cudf::size_type source_begin{9};
  cudf::size_type source_end{size - 50};
  cudf::size_type target_begin{30};
  auto target_end = target_begin + (source_end - source_begin);
  auto row_diff = source_begin - target_begin;

  auto target = cudf::test::fixed_width_column_wrapper<T>(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + size);

  auto source_elements =
    cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return i * 2; });
  auto source = cudf::test::fixed_width_column_wrapper<T>(
    source_elements,
    source_elements + size);

  auto expected_elements =
    cudf::test::make_counting_transform_iterator(
      0,
      [target_begin, target_end, row_diff](auto i) {
        return ((i >= target_begin) && (i < target_end)) ?
          (i + row_diff) * 2 : i;
     });
  auto expected = cudf::test::fixed_width_column_wrapper<T>(
    expected_elements,
    expected_elements + size);

  cudf::mutable_column_view target_view{target};
  this->test(source, expected, target_view,
             source_begin, source_end, target_begin);
}

TYPED_TEST(CopyRangeTypedTestFixture, CopyWithNullsNonzeroOffset)
{
  using T = TypeParam;

  cudf::size_type size{CopyRangeTypedTestFixture<T>::column_size};
  cudf::size_type source_offset{27};
  cudf::size_type source_begin{9};
  cudf::size_type source_end{50};
  cudf::size_type target_offset{58};
  cudf::size_type target_begin{30};
  auto target_end = target_begin + (source_end - source_begin);
  auto row_diff =
    (source_offset + source_begin) - (target_offset + target_begin);

  auto target = cudf::test::fixed_width_column_wrapper<T>(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + size,
    cudf::test::make_counting_transform_iterator(0, all_valid));

  cudf::mutable_column_view tmp = target;
  cudf::mutable_column_view target_slice(tmp.type(), tmp.size() - target_offset,
                                         tmp.head<T>(), tmp.null_mask(),
                                         tmp.null_count(), target_offset);

  auto source_elements =
    cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return i * 2; });
  auto source = cudf::test::fixed_width_column_wrapper<T>(
    source_elements,
    source_elements + size,
    cudf::test::make_counting_transform_iterator(
      0, even_valid));

  auto source_slice = cudf::experimental::slice(
    source,
    std::vector<cudf::size_type>{source_offset, size})[0];

  auto expected_elements =
    cudf::test::make_counting_transform_iterator(
      0,
      [target_offset, target_begin, target_end, row_diff](auto i) {
        return ((i >= target_offset + target_begin) &&
                (i < target_offset + target_end)) ?
          (i + row_diff) * 2 : i;
     });
  auto expected = cudf::test::fixed_width_column_wrapper<T>(
    expected_elements,
    expected_elements + size,
    cudf::test::make_counting_transform_iterator(
      0,
      [target_offset, target_begin, target_end, row_diff](auto i) {
        return ((i >= target_offset + target_begin) &&
                (i < target_offset + target_end)) ?
          even_valid(i + row_diff) : all_valid(i);
      }));

  auto expected_slice =
    cudf::experimental::slice(
      expected,
      std::vector<cudf::size_type>{target_offset, size})[0];

  this->test(source_slice, expected_slice, target_slice,
             source_begin, source_end, target_begin);
}

class CopyRangeErrorTestFixture : public cudf::test::BaseFixture {};

TEST_F(CopyRangeErrorTestFixture, InvalidInplaceCall)
{
  cudf::size_type size{100};

  auto target = cudf::test::fixed_width_column_wrapper<int32_t>(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + size);

  auto source = cudf::test::fixed_width_column_wrapper<int32_t>(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + size,
    cudf::test::make_counting_transform_iterator(0, even_valid));

  cudf::mutable_column_view target_view{target};
  // source has null values but target is not nullable.
  EXPECT_THROW(cudf::experimental::copy_range(source, target_view,
                                              0, size, 0),
               cudf::logic_error);

  std::vector<std::string>
    strings{"", "this", "is", "a", "column", "of", "strings"};
  auto target_string =
    cudf::test::strings_column_wrapper(strings.begin(), strings.end());
  auto source_string =
    cudf::test::strings_column_wrapper(strings.begin(), strings.end());

  cudf::mutable_column_view target_view_string{target_string};
  EXPECT_THROW(cudf::experimental::copy_range(
                 source_string, target_view_string, 0, size, 0),
               cudf::logic_error);
}

TEST_F(CopyRangeErrorTestFixture, InvalidRange)
{
  cudf::size_type size{100};

  auto target = cudf::test::fixed_width_column_wrapper<int32_t>(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + size);

  auto source = cudf::test::fixed_width_column_wrapper<int32_t>(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + size);

  cudf::mutable_column_view target_view{target};

  // empty_range == no-op, this is valid
  EXPECT_NO_THROW(cudf::experimental::copy_range(
                    source, target_view, 0, 0, 0));
  EXPECT_NO_THROW(auto p_ret = cudf::experimental::copy_range(
                    source, target, 0, 0, 0));

  // source_begin is negative
  EXPECT_THROW(cudf::experimental::copy_range(
                 source, target_view, -1, size, 0),
               cudf::logic_error);
  EXPECT_THROW(auto p_ret = cudf::experimental::copy_range(
                 source, target, -1, size, 0),
               cudf::logic_error);

  // source_begin > source_end
  EXPECT_THROW(cudf::experimental::copy_range(
                 source, target_view, 10, 5, 0),
               cudf::logic_error);
  EXPECT_THROW(auto p_ret = cudf::experimental::copy_range(
                 source, target, 10, 5, 0),
               cudf::logic_error);

  // source_begin >= source.size()
  EXPECT_THROW(cudf::experimental::copy_range(
                 source, target_view, 100, 100, 0),
               cudf::logic_error);
  EXPECT_THROW(auto p_ret = cudf::experimental::copy_range(
                 source, target, 100, 100, 0),
               cudf::logic_error);

  // source_end > source.size()
  EXPECT_THROW(cudf::experimental::copy_range(
                 source, target_view, 99, 101, 0),
               cudf::logic_error);
  EXPECT_THROW(auto p_ret = cudf::experimental::copy_range(
                 source, target, 99, 101, 0),
               cudf::logic_error);

  // target_begin < 0
  EXPECT_THROW(cudf::experimental::copy_range(
                 source, target_view, 50, 100, -5),
               cudf::logic_error);
  EXPECT_THROW(auto p_ret = cudf::experimental::copy_range(
                 source, target, 50, 100, -5),
               cudf::logic_error);

  // target_begin >= target.size()
  EXPECT_THROW(cudf::experimental::copy_range(
                 source, target_view, 50, 100, 100),
               cudf::logic_error);
  EXPECT_THROW(auto p_ret = cudf::experimental::copy_range(
                 source, target, 50, 100, 100),
               cudf::logic_error);

  // target_begin + (source_end - source_begin) > target.size()
  EXPECT_THROW(cudf::experimental::copy_range(
                 source, target_view, 50, 100, 80),
               cudf::logic_error);
  EXPECT_THROW(auto p_ret = cudf::experimental::copy_range(
                 source, target, 50, 100, 80),
               cudf::logic_error);
}

TEST_F(CopyRangeErrorTestFixture, DTypeMismatch)
{
  cudf::size_type size{100};

  auto target = cudf::test::fixed_width_column_wrapper<int32_t>(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + size);

  auto source = cudf::test::fixed_width_column_wrapper<float>(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + size);

  cudf::mutable_column_view target_view{target};

  EXPECT_THROW(cudf::experimental::copy_range(
                 source, target_view, 0, 100, 0),
               cudf::logic_error);
  EXPECT_THROW(auto p_ret = cudf::experimental::copy_range(
                 source, target, 0, 100, 0),
               cudf::logic_error);
}

TEST_F(CopyRangeErrorTestFixture, StringCategoryNotSupported)
{
  std::vector<std::string>
    strings{"", "this", "is", "a", "column", "of", "strings"};
  auto target_string =
    cudf::test::strings_column_wrapper(strings.begin(), strings.end());
  auto source_string =
    cudf::test::strings_column_wrapper(strings.begin(), strings.end());

  EXPECT_THROW(auto p_ret = cudf::experimental::copy_range(
                 source_string, target_string, 0, 1, 0),
               cudf::logic_error);
}
