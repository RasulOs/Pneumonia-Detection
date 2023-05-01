-- ANN-CPU BUILD SCRIPT ---

workspace "ann_cpu"
    configurations { "debug", "release" }
    location "build"

project "stb_image"
    kind "StaticLib"
    language "C"
    location "../ThirdParty/build/stb_image"
    targetdir "../ThirdParty/build/%{cfg.buildcfg}"

    includedirs { "../ThirdParty" }

    files { "../ThirdParty/stb_image.c" }

    filter "configurations:debug"
        defines { "_DEBUG", "DEBUG" }
        symbols "On"

    filter "configurations:release"
        defines { "NDEBUG" }
        optimize "On"

project "ann_cpu"
    kind "ConsoleApp"
    language "C++"
    location "build/ann_cpu"
    targetdir "build/%{cfg.buildcfg}"

    includedirs { "../ThirdParty" }

    files { "ann_cpu.cpp", "DataLoader.cpp" }

    links { "stb_image" }

    filter "system:linux"
        links { "pthread", "m" }

    filter "configurations:debug"
        defines { "_DEBUG", "DEBUG" }
        symbols "On"

    filter "configurations:release"
        defines { "NDEBUG" }
        optimize "On"

    filter { "system:linux", "action:gmake2" }
        buildoptions { "-std=c++20",
                       "-march=x86-64",
                       "-Wall",
                       "-Wextra",
                       "-Werror",
                       "-Wpedantic",
                       "-Wfloat-equal",
                       "-Wundef",
                       "-Wshadow",
                       "-Wpointer-arith",
                       "-Wcast-align",
                       "-Wwrite-strings",
                       "-Wswitch-enum",
                       "-Wcast-qual",
                       "-Wconversion",
                       "-Wduplicated-cond",
                       "-Wduplicated-branches",
                       "-Wnon-virtual-dtor",
                       "-Woverloaded-virtual",
                       "-Wold-style-cast",
                       "-Wformat-nonliteral",
                       "-Wformat-security",
                       "-Wformat-y2k",
                       "-Wformat=2",
                       "-Wno-unused-parameter",
                       "-Wunused",
                       "-Wimport",
                       "-Winvalid-pch",
                       "-Wlogical-op",
                       "-Wmissing-declarations",
                       "-Wmissing-field-initializers",
                       "-Wmissing-format-attribute",
                       "-Wmissing-include-dirs",
                       "-Wmissing-noreturn",
                       "-Wpacked",
                       "-Wredundant-decls",
                       "-Wstack-protector",
                       "-Wstrict-null-sentinel",
                       "-Wdisabled-optimization",
                       "-Wsign-conversion",
                       "-Wsign-promo",
                       "-Wstrict-aliasing=2",
                       "-Wstrict-overflow=2",
                       "-fno-rtti",
                       "-fno-exceptions",
                       "-Wno-suggest-attribute=format" }

    filter { "system:linux", "action:gmake2", "configurations:debug" }
        buildoptions { "-Wno-unused-but-set-variable",
                       "-Wno-unused-variable",
                       "-Wno-unused-function",
                       "-fno-omit-frame-pointer" }
