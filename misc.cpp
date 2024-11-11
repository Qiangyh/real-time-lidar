#include "misc.h"
#include <iostream>
#include <boost/lexical_cast.hpp>

bool askYesNo(const std::string &str)
{
	bool set = false;
	bool ret;
	std::cout << str << " [y/n]: ";
	while (!set)
	{
		std::string input;
		std::getline(std::cin, input);
		if (input.compare("y") == 0)
		{
			set = true;
			ret = true;
		}
		else if (input.compare("n") == 0)
		{
			set = true;
			ret = false;
		}
		else
		{
			std::cout << "Wrong Input. Answer [y/n]: ";
		}
	}
	return ret;
}

int ask_for_param(std::string message, int min, int max, int defaultValue)
{

	std::cout << message << "(default: " << defaultValue << "): ";
	std::string s;

	bool flag = false;
	int ret;

	while (!flag)
	{
		std::getline(std::cin, s);
		if (s.empty())
		{
			ret = defaultValue;
			flag = true;
		}
		else
		{
			try
			{
				ret = boost::lexical_cast<int>(s);
				if (ret < min || ret > max)
				{
					std::cout << "Wrong Input, try between " << min << " and " << max << std::endl;
				}
				else
					flag = true;
			}
			catch (boost::bad_lexical_cast)
			{
				std::cout << "Wrong Input, try between " << min << " and " << max << std::endl;
			}
		}
	}

	std::cout << "Value chosen: " << ret << std::endl;
	return ret;
}

float ask_for_paramf(std::string message, float min, float max, float defaultValue)
{

	std::cout << message << "(default: " << defaultValue << "): ";
	std::string s;

	bool flag = false;
	float ret;

	while (!flag)
	{
		std::getline(std::cin, s);
		if (s.empty())
		{
			ret = defaultValue;
			flag = true;
		}
		else
		{
			try
			{
				ret = boost::lexical_cast<float>(s);
				if (ret < min || ret > max)
				{
					std::cout << "Wrong Input, try between " << min << " and " << max << std::endl;
				}
				else
					flag = true;
			}
			catch (boost::bad_lexical_cast)
			{
				std::cout << "Wrong Input, try between " << min << " and " << max << std::endl;
			}
		}
	}

	std::cout << "Value chosen: " << ret << std::endl;
	return ret;
}