// Preprocessor: A tool to pre-process images in the pneumonia dataset
// Copyright (C) 2023 saccharineboi
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <thread>
#include <filesystem>

#include <cstdio>
#include <cstring>

////////////////////////////////////////
#define PREPROCESSOR_VERSION_MAJOR 0
#define PREPROCESSOR_VERSION_MINOR 1

////////////////////////////////////////
struct Config
{
    std::string rootDirPath{ "." };
    bool quiet{};
};

////////////////////////////////////////////////////////////////////////////////////////////////////
static void ParseArgAssert(int argc, int i, const char* argName)
{
    if (argc - 1 == i)
    {
        std::fprintf(stderr, "ERROR: value for %s wasn't given\n", argName);
        std::exit(EXIT_FAILURE);
    }
}

////////////////////////////////////////
static Config ParseArgs(int argc, char** argv)
{
    Config config;
    for (int i = 1; i < argc; ++i)
    {
        ////////////////////////////////////////
        /////// OPTIONS THAT TERMINATE /////////
        ////////////////////////////////////////

        if (!std::strncmp("--help", argv[i], 50))
        {
            std::printf("Usage: %s [OPTIONS...]\n\n", argv[0]);
            std::printf("Options:\n");
            std::printf("\t--help:\t\t Print this help message\n");
            std::printf("\t--license:\t Print the license\n");
            std::printf("\t--version:\t Print the version number\n");
            std::printf("\t--quiet:\t Don't write to stdout\n");
            std::printf("\t--root-dir:\t Directory containing the dataset\n");

            std::exit(EXIT_SUCCESS);
        }
        else if (!std::strncmp("--license", argv[i], 50))
        {
#define LICENSE "Copyright (C) 2023 saccharineboi\n" \
                "This program is free software: you can redistribute it and/or modify\n" \
                "it under the terms of the GNU General Public License as published by\n" \
                "the Free Software Foundation, either version 3 of the License, or\n" \
                "(at your option) any later version.\n\n" \
                "This program is distributed in the hope that it will be useful,\n" \
                "but WITHOUT ANY WARRANTY; without even the implied warranty of\n" \
                "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n" \
                "GNU General Public License for more details.\n\n" \
                "You should have received a copy of the GNU General Public License\n" \
                "along with this program.  If not, see <https://www.gnu.org/licenses/>.\n"

                std::printf("%s", LICENSE);
                std::exit(EXIT_SUCCESS);
        }
        else if (!std::strncmp("--version", argv[i], 50))
        {
            std::printf("%d.%d\n", PREPROCESSOR_VERSION_MAJOR, PREPROCESSOR_VERSION_MINOR);
            std::exit(EXIT_SUCCESS);
        }

        ////////////////////////////////////////
        ///// OPTIONS THAT ENABLE/DISABLE //////
        ////////////////////////////////////////

        if (!std::strncmp("--quiet", argv[i], 50))
        {
            config.quiet = true;
        }

        ////////////////////////////////////////
        ///////// OPTIONS WITH VALUES //////////
        ////////////////////////////////////////

        if (!std::strncmp("--root-dir", argv[i], 50))
        {
            ParseArgAssert(argc, i, "--root-dir");
            config.rootDirPath = argv[i + 1];
        }
    }
    return config;
}

////////////////////////////////////////
int main(int argc, char** argv)
{
    Config config = ParseArgs(argc, argv);

    if (!config.quiet)
    {
        std::printf("Root Directory: %s\n", config.rootDirPath.c_str());
    }

    return 0;
}
