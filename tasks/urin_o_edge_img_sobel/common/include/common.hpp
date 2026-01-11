#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace urin_o_edge_img_sobel {

// Входной тип: tuple из пикселей, высоты и ширины
// Пиксели — одномерный массив int (оттенки серого или цвет как набор int)
using InType = std::tuple<std::vector<int>, int, int>;

// Выходной тип: одномерный массив int — результат после фильтра Собеля
using OutType = std::vector<int>;

// Тип для тестов — например, тестовый кейс с ID и описанием
using TestType = std::tuple<int, std::string>;

// Базовый таск, шаблон из task.hpp
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace urin_o_edge_img_sobel
