#ifndef _CONNECTIONS_H_
#define _CONNECTIONS_H_
#include "deep_layer.h"

const int l1_l2_connections[][10] = {

{	39, 6, 41, 51, 17, 63, 10, 44, 41, 13 	},
{	58, 43, 50, 59, 35, 6, 60, 2, 20, 56 	},
{	27, 40, 39, 13, 54, 26, 46, 35, 51, 31 	},
{	9, 26, 38, 50, 13, 55, 49, 24, 35, 26 	},
{	37, 29, 5, 23, 24, 41, 30, 20, 43, 50 	},
{	13, 6, 27, 52, 20, 17, 14, 2, 52, 1 	},
{	33, 61, 28, 7, 48, 41, 62, 33, 1, 33 	},
{	60, 39, 62, 1, 62, 23, 42, 28, 43, 22 	},
{	15, 56, 28, 42, 44, 48, 59, 59, 50, 47 	},
{	60, 20, 44, 24, 27, 28, 2, 26, 62, 3 	},
{	59, 58, 42, 58, 59, 41, 17, 38, 5, 60 	},
{	60, 20, 53, 24, 62, 33, 9, 57, 28, 59 	},
{	40, 25, 15, 21, 49, 43, 49, 51, 5, 47 	},
{	55, 0, 41, 33, 58, 37, 10, 11, 11, 16 	},
{	8, 7, 36, 61, 31, 35, 30, 40, 28, 59 	},
{	36, 5, 20, 51, 26, 5, 30, 11, 57, 35 	},
{	59, 48, 36, 36, 17, 30, 9, 28, 42, 20 	},
{	44, 50, 27, 16, 47, 59, 51, 13, 35, 16 	},
{	8, 7, 21, 28, 59, 47, 34, 25, 58, 27 	},
{	61, 53, 11, 33, 26, 28, 63, 35, 56, 41 	},
{	56, 36, 27, 19, 53, 10, 14, 40, 24, 50 	},
{	56, 32, 57, 13, 61, 52, 60, 31, 14, 55 	},
{	58, 11, 44, 5, 44, 6, 33, 43, 42, 26 	},
{	21, 34, 62, 48, 53, 51, 59, 4, 28, 19 	},
{	54, 20, 51, 47, 34, 48, 36, 30, 15, 50 	},
{	21, 9, 61, 2, 14, 41, 8, 48, 20, 50 	},
{	10, 41, 20, 8, 26, 10, 60, 21, 14, 24 	},
{	40, 4, 44, 27, 51, 14, 12, 23, 45, 27 	},
{	9, 2, 37, 6, 4, 51, 47, 13, 35, 4 	},
{	63, 45, 45, 20, 54, 7, 30, 50, 28, 44 	},
{	10, 4, 48, 54, 32, 35, 5, 44, 59, 50 	},
{	7, 4, 52, 44, 11, 57, 32, 58, 6, 3 	},
{	62, 5, 49, 44, 25, 39, 51, 55, 25, 16 	},
{	35, 35, 20, 19, 25, 52, 55, 30, 32, 50 	},
{	16, 40, 54, 5, 20, 1, 62, 52, 60, 4 	},
{	56, 58, 9, 41, 38, 35, 16, 26, 26, 41 	},
{	42, 62, 12, 62, 17, 37, 51, 8, 4, 19 	},
{	58, 20, 59, 49, 25, 16, 50, 23, 4, 46 	},
{	27, 60, 41, 37, 37, 15, 8, 53, 41, 34 	},
{	30, 19, 32, 42, 18, 50, 16, 5, 58, 20 	},
{	24, 53, 40, 20, 38, 2, 36, 24, 25, 40 	},
{	7, 53, 37, 48, 26, 10, 63, 34, 0, 41 	},
{	4, 30, 60, 37, 9, 14, 23, 25, 19, 17 	},
{	45, 44, 6, 21, 0, 44, 23, 36, 5, 49 	},
{	12, 12, 38, 49, 60, 0, 60, 59, 34, 60 	},
{	36, 38, 26, 33, 11, 35, 47, 34, 60, 3 	},
{	52, 41, 47, 58, 63, 47, 39, 22, 19, 44 	},
{	7, 31, 56, 45, 17, 52, 45, 13, 47, 15 	},
{	9, 20, 54, 35, 53, 1, 7, 36, 36, 3 	},
{	39, 24, 45, 22, 18, 44, 5, 57, 2, 24 	},
{	37, 10, 56, 29, 55, 9, 17, 37, 22, 1 	},
{	52, 31, 21, 42, 2, 10, 44, 9, 46, 16 	},
{	13, 22, 40, 58, 44, 58, 38, 50, 52, 40 	},
{	10, 25, 50, 2, 55, 42, 11, 8, 15, 33 	},
{	9, 3, 0, 30, 46, 3, 40, 26, 12, 23 	},
{	42, 25, 45, 18, 19, 25, 12, 57, 11, 0 	},
{	34, 22, 26, 20, 24, 17, 62, 36, 25, 13 	},
{	5, 35, 17, 6, 1, 63, 9, 42, 25, 21 	},
{	1, 3, 47, 46, 21, 2, 7, 33, 60, 19 	},
{	34, 30, 41, 60, 50, 1, 13, 49, 37, 38 	},
{	62, 43, 9, 15, 49, 11, 14, 58, 53, 39 	},
{	15, 54, 42, 62, 36, 63, 1, 43, 33, 61 	},
{	62, 3, 27, 39, 63, 13, 41, 12, 62, 14 	},
{	50, 61, 57, 60, 12, 42, 7, 27, 36, 60 	},
{	2, 52, 50, 45, 50, 22, 44, 51, 1, 13 	},
{	48, 0, 16, 11, 39, 15, 25, 16, 27, 23 	},
{	31, 14, 20, 24, 10, 33, 3, 17, 60, 39 	},
{	13, 62, 27, 63, 43, 14, 21, 24, 1, 22 	},
{	37, 50, 22, 54, 61, 62, 5, 22, 14, 33 	},
{	46, 45, 47, 2, 6, 57, 35, 9, 10, 31 	},
{	48, 23, 30, 12, 22, 9, 26, 43, 33, 27 	},
{	1, 7, 13, 24, 61, 11, 22, 2, 33, 36 	},
{	35, 15, 18, 18, 18, 24, 11, 53, 33, 21 	},
{	21, 17, 44, 51, 29, 2, 60, 55, 45, 30 	},
{	19, 47, 37, 32, 7, 34, 43, 29, 36, 13 	},
{	1, 8, 28, 19, 26, 46, 43, 38, 36, 12 	},
{	59, 57, 30, 40, 44, 59, 42, 40, 51, 24 	},
{	6, 6, 7, 43, 38, 14, 13, 18, 43, 50 	},
{	31, 44, 58, 59, 0, 20, 42, 43, 58, 14 	},
{	56, 54, 7, 22, 30, 51, 17, 8, 27, 4 	},
{	32, 34, 10, 39, 13, 49, 53, 27, 3, 32 	},
{	13, 34, 13, 7, 29, 13, 27, 7, 56, 22 	},
{	21, 48, 12, 28, 6, 42, 15, 24, 50, 43 	},
{	28, 19, 13, 39, 58, 26, 24, 48, 53, 27 	},
{	16, 2, 61, 29, 9, 26, 42, 37, 34, 35 	},
{	59, 55, 19, 7, 20, 26, 49, 35, 50, 35 	},
{	14, 14, 54, 27, 53, 49, 54, 13, 33, 43 	},
{	40, 49, 46, 37, 15, 55, 0, 57, 28, 34 	},
{	28, 23, 25, 48, 30, 45, 10, 15, 17, 60 	},
{	51, 31, 10, 41, 59, 0, 26, 49, 13, 59 	},
{	28, 54, 45, 10, 27, 60, 2, 27, 53, 30 	},
{	61, 18, 54, 23, 2, 20, 4, 12, 36, 21 	},
{	8, 23, 53, 18, 0, 48, 18, 27, 33, 32 	},
{	22, 61, 22, 3, 8, 49, 63, 10, 13, 53 	},
{	40, 10, 7, 30, 33, 9, 51, 38, 21, 23 	},
{	59, 29, 46, 48, 47, 46, 32, 2, 9, 1 	},
{	34, 32, 63, 56, 35, 7, 41, 35, 17, 54 	},
{	24, 57, 1, 31, 24, 34, 40, 11, 8, 61 	},
{	34, 4, 26, 16, 52, 9, 62, 21, 11, 8 	},
{	22, 45, 40, 21, 37, 11, 28, 15, 46, 45 	},
{	5, 6, 39, 6, 37, 63, 41, 13, 10, 49 	},
{	10, 44, 53, 36, 60, 42, 46, 58, 63, 57 	},
{	2, 21, 39, 42, 43, 12, 54, 7, 27, 36 	},
{	53, 33, 43, 28, 39, 16, 27, 16, 30, 37 	},
{	2, 40, 17, 55, 13, 13, 33, 59, 7, 32 	},
{	52, 10, 54, 27, 52, 33, 40, 42, 40, 3 	},
{	15, 29, 36, 58, 57, 12, 10, 20, 28, 40 	},
{	57, 30, 17, 10, 22, 30, 23, 55, 25, 31 	},
{	24, 13, 41, 14, 41, 29, 47, 17, 8, 23 	},
{	20, 23, 53, 57, 17, 46, 5, 27, 3, 33 	},
{	4, 60, 0, 21, 7, 22, 51, 30, 13, 12 	},
{	61, 37, 25, 38, 51, 2, 4, 34, 19, 12 	},
{	58, 40, 35, 47, 33, 52, 29, 38, 15, 32 	},
{	7, 19, 29, 7, 40, 36, 29, 27, 2, 43 	},
{	39, 0, 16, 1, 38, 4, 3, 42, 38, 23 	},
{	54, 32, 63, 25, 15, 32, 13, 45, 6, 29 	},
{	13, 13, 48, 42, 21, 25, 14, 50, 52, 17 	},
{	29, 28, 17, 46, 29, 55, 50, 32, 34, 24 	},
{	55, 24, 57, 54, 50, 8, 22, 63, 53, 28 	},
{	28, 3, 42, 13, 45, 63, 38, 60, 49, 26 	},
{	13, 15, 54, 30, 61, 19, 21, 47, 52, 55 	},
{	7, 43, 16, 0, 34, 2, 9, 56, 1, 62 	},
{	21, 30, 1, 63, 43, 47, 62, 17, 43, 47 	},
{	43, 56, 62, 34, 22, 59, 53, 43, 42, 41 	},
{	35, 50, 21, 51, 50, 55, 53, 59, 47, 54 	},
{	58, 4, 20, 59, 3, 63, 42, 1, 16, 21 	},
{	49, 60, 13, 47, 30, 35, 43, 19, 15, 21 	},
{	61, 50, 7, 18, 37, 58, 9, 26, 53, 56 	},


};

const int l2_l3_connections[][8] = {
{	103, 70, 105, 115, 81, 127, 74, 108 	},
{	41, 77, 58, 43, 114, 123, 99, 70 	},
{	124, 66, 84, 120, 27, 104, 103, 13 	},
{	118, 90, 46, 99, 51, 31, 73, 26 	},
{	102, 50, 13, 55, 49, 88, 35, 90 	},
{	37, 93, 5, 23, 88, 105, 94, 84 	},
{	43, 50, 77, 70, 27, 52, 84, 17 	},
{	14, 2, 116, 65, 33, 61, 92, 7 	},
{	112, 105, 62, 33, 65, 97, 124, 103 	},
{	62, 1, 126, 23, 106, 92, 107, 22 	},
{	15, 56, 92, 42, 108, 48, 59, 123 	},
{	50, 47, 60, 84, 108, 24, 91, 92 	},
{	2, 26, 126, 67, 123, 122, 42, 58 	},
{	123, 41, 81, 102, 5, 60, 124, 20 	},
{	117, 88, 62, 97, 9, 121, 92, 59 	},
{	40, 25, 15, 21, 49, 107, 113, 51 	},
{	5, 111, 119, 0, 105, 33, 58, 101 	},
{	74, 11, 75, 80, 72, 71, 100, 61 	},
{	31, 35, 30, 40, 28, 123, 100, 69 	},
{	20, 115, 90, 69, 94, 75, 121, 99 	},
{	59, 112, 100, 36, 17, 30, 9, 92 	},
{	42, 84, 44, 114, 27, 16, 47, 59 	},
{	51, 77, 99, 80, 72, 71, 21, 92 	},
{	59, 111, 34, 25, 58, 27, 125, 117 	},
{	11, 97, 26, 28, 127, 35, 120, 41 	},
{	120, 36, 27, 19, 53, 74, 78, 104 	},
{	24, 50, 56, 96, 121, 77, 61, 52 	},
{	60, 95, 78, 119, 122, 75, 108, 5 	},
{	44, 6, 33, 43, 42, 26, 85, 34 	},
{	62, 112, 53, 115, 59, 4, 92, 83 	},
{	54, 20, 51, 47, 98, 112, 100, 30 	},
{	79, 50, 21, 73, 125, 2, 78, 41 	},
{	8, 112, 84, 50, 10, 41, 84, 72 	},
{	26, 10, 60, 85, 14, 24, 40, 68 	},
{	44, 91, 115, 14, 76, 87, 45, 27 	},
{	9, 66, 101, 6, 68, 51, 47, 77 	},
{	35, 4, 127, 45, 45, 84, 118, 71 	},
{	94, 50, 28, 108, 74, 68, 48, 118 	},
{	32, 35, 5, 108, 123, 50, 7, 4 	},
{	116, 108, 11, 57, 32, 58, 6, 67 	},
{	62, 5, 113, 108, 89, 103, 51, 55 	},
{	25, 80, 35, 99, 20, 83, 89, 52 	},
{	119, 94, 32, 114, 16, 40, 118, 5 	},
{	20, 1, 62, 52, 60, 68, 120, 122 	},
{	73, 105, 102, 35, 80, 26, 90, 105 	},
{	106, 126, 76, 126, 81, 37, 51, 72 	},
{	4, 83, 58, 20, 123, 49, 25, 16 	},
{	50, 87, 68, 110, 27, 60, 105, 101 	},
{	37, 79, 8, 117, 105, 98, 94, 83 	},
{	96, 42, 82, 50, 80, 5, 122, 84 	},
{	88, 53, 104, 84, 102, 2, 100, 24 	},
{	89, 40, 7, 117, 101, 112, 90, 10 	},
{	63, 98, 0, 41, 68, 94, 124, 37 	},
{	9, 78, 87, 89, 83, 81, 45, 44 	},
{	6, 21, 0, 108, 23, 100, 5, 113 	},
{	12, 12, 102, 113, 124, 64, 124, 59 	},
{	34, 124, 100, 102, 90, 97, 11, 99 	},
{	47, 98, 60, 3, 52, 105, 47, 58 	},
{	127, 47, 39, 22, 19, 44, 7, 31 	},
{	56, 109, 17, 52, 45, 13, 111, 79 	},
{	9, 84, 54, 99, 53, 65, 71, 100 	},
{	36, 3, 103, 88, 109, 22, 18, 108 	},
{	69, 57, 2, 88, 101, 10, 120, 29 	},
{	119, 9, 81, 37, 22, 65, 116, 31 	},
{	21, 42, 2, 74, 108, 73, 46, 16 	},
{	77, 22, 104, 58, 44, 122, 38, 114 	},
{	52, 40, 74, 25, 50, 66, 55, 42 	},
{	75, 8, 79, 97, 73, 67, 0, 94 	},
{	110, 3, 40, 90, 76, 87, 106, 25 	},
{	109, 82, 83, 25, 76, 121, 11, 0 	},
{	34, 86, 26, 84, 24, 81, 126, 100 	},
{	89, 77, 69, 35, 17, 70, 1, 127 	},
{	73, 42, 89, 21, 1, 67, 47, 110 	},
{	21, 2, 7, 97, 124, 19, 98, 30 	},
{	105, 124, 114, 1, 77, 113, 101, 38 	},
{	62, 43, 73, 79, 113, 75, 78, 58 	},
{	117, 39, 79, 118, 106, 126, 100, 127 	},
{	1, 107, 97, 125, 126, 67, 27, 103 	},
{	63, 13, 105, 12, 126, 78, 50, 61 	},
{	121, 124, 12, 106, 71, 91, 36, 60 	},
{	2, 116, 50, 109, 114, 22, 108, 115 	},
{	1, 77, 112, 0, 16, 11, 103, 79 	},
{	25, 80, 91, 23, 31, 14, 84, 24 	},
{	10, 97, 3, 81, 60, 39, 13, 62 	},
{	27, 63, 43, 14, 85, 24, 1, 86 	},
{	101, 114, 86, 118, 125, 62, 69, 22 	},
{	14, 33, 46, 45, 47, 2, 70, 57 	},
{	99, 73, 10, 31, 112, 23, 94, 12 	},
{	86, 9, 26, 43, 33, 27, 1, 7 	},
{	13, 88, 125, 11, 22, 66, 33, 36 	},
{	99, 79, 82, 18, 82, 24, 75, 53 	},
{	97, 85, 85, 81, 108, 51, 93, 66 	},
{	60, 119, 109, 94, 19, 111, 101, 32 	},
{	71, 98, 43, 93, 36, 77, 1, 8 	},
{	28, 83, 26, 110, 107, 102, 36, 76 	},
{	59, 121, 30, 40, 44, 123, 106, 104 	},
{	115, 88, 70, 6, 71, 43, 38, 14 	},
{	13, 82, 107, 50, 31, 108, 58, 59 	},
{	64, 84, 42, 43, 58, 78, 120, 118 	},
{	71, 22, 30, 115, 17, 8, 91, 4 	},
{	96, 34, 10, 39, 77, 49, 53, 91 	},
{	3, 32, 13, 34, 13, 71, 93, 77 	},
{	27, 7, 120, 86, 85, 112, 76, 28 	},
{	6, 106, 15, 24, 114, 107, 28, 83 	},
{	13, 39, 122, 90, 88, 48, 53, 91 	},
{	80, 66, 125, 93, 9, 90, 42, 37 	},
{	98, 35, 123, 55, 19, 71, 84, 26 	},
{	49, 99, 50, 35, 78, 78, 118, 91 	},
{	117, 113, 54, 77, 33, 107, 40, 113 	},
{	46, 37, 79, 55, 0, 121, 92, 98 	},
{	28, 87, 25, 48, 30, 109, 74, 79 	},
{	81, 124, 115, 31, 74, 105, 123, 64 	},
{	90, 49, 13, 123, 28, 54, 109, 74 	},
{	91, 60, 2, 91, 53, 94, 61, 82 	},
{	54, 87, 2, 84, 68, 76, 36, 21 	},
{	72, 23, 53, 18, 0, 48, 82, 91 	},
{	97, 96, 86, 125, 22, 67, 72, 113 	},
{	127, 74, 77, 53, 40, 10, 7, 94 	},
{	97, 9, 51, 38, 85, 87, 59, 29 	},
{	110, 112, 47, 110, 32, 2, 73, 1 	},
{	98, 32, 127, 120, 99, 71, 105, 99 	},
{	17, 54, 24, 57, 65, 31, 24, 34 	},
{	40, 75, 72, 125, 34, 4, 26, 16 	},
{	116, 73, 126, 21, 75, 72, 22, 45 	},
{	104, 21, 37, 75, 92, 15, 46, 109 	},
{	69, 70, 39, 6, 101, 63, 41, 13 	},
{	10, 113, 10, 44, 117, 36, 60, 106 	},
{	110, 58, 127, 57, 2, 21, 103, 106 	},
{	43, 12, 54, 7, 27, 100, 117, 97 	},
{	43, 28, 103, 16, 91, 16, 30, 101 	},
{	2, 40, 17, 119, 77, 77, 97, 59 	},
{	7, 96, 116, 10, 118, 91, 116, 33 	},
{	104, 42, 40, 3, 15, 29, 100, 58 	},
{	57, 76, 74, 20, 92, 104, 121, 94 	},
{	17, 10, 86, 94, 87, 55, 25, 95 	},
{	24, 13, 105, 14, 105, 93, 47, 81 	},
{	8, 87, 84, 23, 117, 57, 81, 46 	},
{	5, 27, 67, 97, 4, 60, 64, 21 	},
{	71, 22, 115, 30, 77, 12, 125, 101 	},
{	25, 102, 115, 2, 68, 34, 83, 76 	},
{	122, 40, 99, 111, 97, 52, 29, 102 	},
{	79, 96, 71, 83, 29, 7, 104, 100 	},
{	29, 91, 2, 107, 103, 0, 80, 1 	},
{	102, 68, 3, 42, 102, 87, 118, 96 	},
{	127, 89, 79, 96, 13, 109, 70, 93 	},
{	77, 13, 48, 106, 21, 25, 78, 50 	},
{	116, 81, 29, 92, 81, 110, 93, 55 	},
{	50, 96, 98, 24, 55, 88, 121, 54 	},
{	50, 72, 22, 63, 53, 92, 28, 3 	},
{	106, 77, 109, 127, 102, 60, 49, 90 	},
{	13, 79, 54, 94, 61, 19, 21, 111 	},
{	116, 119, 7, 43, 80, 0, 98, 2 	},
{	73, 120, 65, 126, 85, 94, 1, 63 	},
{	43, 111, 62, 17, 43, 111, 107, 56 	},
{	62, 34, 22, 123, 53, 43, 106, 41 	},
{	35, 114, 85, 115, 114, 55, 117, 59 	},
{	47, 54, 58, 4, 20, 59, 67, 63 	},
{	42, 1, 80, 85, 113, 60, 13, 47 	},
{	94, 35, 43, 19, 79, 21, 61, 114 	},
{	7, 18, 101, 122, 73, 90, 53, 120 	},
{	16, 111, 125, 37, 43, 64, 100, 85 	},
{	66, 53, 43, 51, 113, 56, 98, 79 	},
{	92, 13, 98, 43, 35, 31, 29, 42 	},
{	49, 2, 36, 122, 92, 90, 115, 108 	},
{	73, 112, 17, 116, 48, 118, 74, 114 	},
{	43, 117, 37, 28, 45, 8, 107, 9 	},
{	21, 77, 52, 56, 109, 81, 99, 30 	},
{	83, 7, 25, 47, 97, 12, 28, 43 	},
{	124, 45, 31, 44, 35, 105, 31, 78 	},
{	94, 68, 106, 12, 76, 85, 21, 98 	},
{	35, 74, 26, 16, 27, 125, 46, 111 	},
{	5, 71, 30, 102, 83, 58, 17, 79 	},
{	104, 49, 124, 11, 26, 27, 90, 121 	},
{	95, 68, 5, 44, 26, 26, 14, 61 	},
{	100, 40, 77, 0, 38, 123, 111, 43 	},
{	67, 13, 17, 22, 72, 35, 102, 48 	},
{	84, 98, 59, 110, 125, 21, 103, 92 	},
{	90, 108, 8, 116, 7, 22, 49, 107 	},
{	63, 126, 107, 101, 121, 90, 16, 60 	},
{	104, 33, 83, 48, 68, 57, 96, 24 	},
{	27, 27, 7, 24, 49, 110, 116, 11 	},
{	91, 125, 127, 98, 19, 48, 77, 82 	},
{	46, 57, 55, 39, 19, 71, 100, 123 	},
{	105, 55, 43, 45, 112, 11, 70, 11 	},
{	39, 77, 35, 88, 59, 23, 99, 22 	},
{	20, 98, 120, 40, 18, 70, 122, 64 	},
{	127, 50, 103, 18, 121, 75, 14, 98 	},
{	2, 57, 16, 114, 69, 86, 125, 108 	},
{	35, 32, 68, 94, 56, 39, 117, 76 	},
{	9, 109, 116, 27, 51, 111, 91, 50 	},
{	33, 66, 69, 26, 14, 83, 125, 16 	},
{	12, 13, 3, 81, 99, 0, 61, 6 	},
{	33, 1, 100, 89, 40, 89, 37, 49 	},
{	71, 26, 76, 122, 9, 39, 45, 42 	},
{	106, 114, 68, 120, 69, 65, 8, 81 	},
{	78, 11, 35, 49, 12, 96, 55, 45 	},
{	98, 28, 6, 10, 117, 43, 60, 60 	},
{	69, 8, 55, 78, 48, 100, 120, 26 	},
{	86, 61, 18, 27, 126, 26, 108, 77 	},
{	38, 15, 126, 50, 112, 54, 95, 82 	},
{	82, 101, 92, 71, 16, 24, 4, 86 	},
{	33, 59, 36, 81, 31, 29, 107, 117 	},
{	90, 125, 16, 88, 23, 124, 37, 61 	},
{	12, 36, 111, 124, 90, 78, 78, 44 	},
{	51, 42, 115, 68, 67, 119, 26, 100 	},
{	50, 62, 53, 81, 91, 32, 70, 53 	},
{	29, 86, 14, 52, 83, 51, 114, 95 	},
{	87, 97, 91, 49, 48, 41, 93, 99 	},
{	83, 81, 39, 22, 72, 65, 122, 123 	},
{	0, 47, 76, 91, 79, 19, 17, 108 	},
{	105, 31, 33, 60, 82, 19, 27, 42 	},
{	116, 118, 91, 36, 31, 57, 8, 115 	},
{	10, 47, 9, 82, 113, 4, 77, 113 	},
{	51, 26, 76, 3, 45, 93, 111, 22 	},
{	124, 16, 83, 79, 35, 110, 121, 24 	},
{	101, 84, 60, 4, 13, 68, 119, 23 	},
{	116, 1, 106, 101, 5, 55, 86, 56 	},
{	81, 34, 59, 126, 0, 43, 21, 124 	},
{	59, 104, 75, 95, 86, 68, 119, 59 	},
{	25, 51, 64, 38, 120, 55, 62, 108 	},
{	56, 40, 81, 61, 95, 39, 118, 49 	},
{	73, 49, 47, 73, 92, 68, 70, 24 	},
{	44, 17, 119, 3, 86, 110, 62, 111 	},
{	33, 126, 21, 25, 54, 83, 5, 110 	},
{	123, 86, 44, 91, 125, 34, 12, 71 	},
{	83, 59, 16, 48, 0, 86, 72, 44 	},
{	104, 63, 47, 62, 45, 110, 45, 78 	},
{	108, 66, 104, 34, 22, 109, 17, 17 	},
{	68, 61, 108, 65, 95, 120, 8, 50 	},
{	52, 25, 98, 52, 111, 42, 96, 87 	},
{	105, 16, 21, 22, 126, 66, 101, 106 	},
{	5, 77, 13, 27, 58, 30, 44, 126 	},
{	91, 25, 64, 58, 17, 72, 108, 69 	},
{	97, 79, 121, 81, 121, 90, 40, 99 	},
{	106, 62, 121, 104, 0, 94, 82, 5 	},
{	43, 95, 32, 102, 125, 77, 100, 88 	},
{	102, 36, 18, 119, 109, 127, 61, 78 	},
{	78, 54, 31, 71, 16, 72, 42, 122 	},
{	6, 36, 98, 6, 2, 53, 12, 46 	},
{	20, 44, 20, 18, 121, 120, 106, 95 	},
{	29, 125, 87, 10, 124, 20, 88, 74 	},
{	74, 120, 17, 91, 64, 60, 85, 70 	},
{	96, 56, 76, 98, 109, 88, 16, 1 	},
{	5, 36, 19, 126, 29, 126, 94, 58 	},
{	123, 53, 68, 119, 73, 28, 65, 19 	},
{	20, 82, 110, 84, 14, 68, 26, 110 	},
{	124, 103, 81, 105, 63, 97, 106, 68 	},
{	6, 126, 67, 35, 124, 33, 93, 119 	},
{	86, 33, 110, 31, 61, 47, 50, 82 	},
{	1, 33, 38, 16, 101, 65, 126, 97 	},
{	40, 79, 74, 103, 49, 52, 44, 55 	},
{	50, 111, 90, 46, 16, 55, 37, 102 	},
{	88, 19, 5, 21, 66, 55, 103, 68 	},
{	88, 14, 84, 61, 79, 82, 30, 119 	},
{	34, 104, 94, 83, 29, 10, 10, 79 	},
{	121, 100, 126, 9, 27, 35, 111, 115 	}
};


#endif //
