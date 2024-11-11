#pragma once

#include <vector>
#include <string>

template <typename T>
std::vector<T> linspace(T start, T end, int N)
{
	std::vector<T> ret(N);
	T val = start;

	if (N == 1)
	{
		ret[0] = end;
	}
	else
	{
		T delta = (end - start) / T(N - 1);
		for (int j = 0; j < N; j++)
		{
			ret[j] = val;
			val += delta;
		}
	}
	return ret;
};

template <typename T>
std::vector<T> logspace(T start, T end, int N)
{
	std::vector<T> ret(N);
	T val = start;

	if (N == 1)
	{
		ret[0] = end;
	}
	else
	{
		T delta = (log(end) - log(start)) / T(N - 1);
		for (int j = 0; j < N; j++)
		{
			ret[j] = val;
			val *= exp(delta);
		}
	}
	return ret;
};

int ask_for_param(std::string message, int min, int max, int defaultValue);

float ask_for_paramf(std::string message, float min, float max, float defaultValue);

bool askYesNo(const std::string &str);