/*
 * Copyright (C) 2025 Jacob Davis
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * 
 * The purpose of this program is to control profiling
 * different frameworks for a cubic interpolation.
 */
#ifndef CUBIC_INTERP_CL_H
#define CUBIC_INTERP_CL_H
#ifdef __cplusplus
extern "C" {
#endif
#include "cubic_interp.h"
#include "../main.h"
void test_all_cubic_cl(double **values, FILE *fp);
#ifdef __cplusplus
}
#endif
#endif

