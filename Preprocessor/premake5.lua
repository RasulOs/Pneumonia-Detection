-- PREPROCESSOR BUILD SCRIPT ---

workspace "preprocessor"
    configurations { "debug", "release" }
    location "build"

project "stb_image"
    kind "StaticLib"
    language "C"
    location "../ThirdParty/build/stb_image"
    targetdir "../ThirdParty/build/%{cfg.buildcfg}"

    includedirs { "../ThirdParty" }

    files { "../ThirdParty/stb_image.c" }

    filter "configurations:Debug"
        defines { "_DEBUG", "DEBUG" }
        symbols "On"

    filter "configurations:Release"
        defines { "NDEBUG" }
        optimize "On"

project "stb_image_resize"
    kind "StaticLib"
    language "C"
    location "../ThirdParty/build/stb_image_resize"
    targetdir "../ThirdParty/build/%{cfg.buildcfg}"

    includedirs { "../ThirdParty" }

    files { "../ThirdParty/stb_image_resize.c" }

    filter "configurations:Debug"
        defines { "_DEBUG", "DEBUG" }
        symbols "On"

    filter "configurations:Release"
        defines { "NDEBUG" }
        optimize "On"

project "stb_image_write"
    kind "StaticLib"
    language "C"
    location "../ThirdParty/build/stb_image_write"
    targetdir "../ThirdParty/build/%{cfg.buildcfg}"

    includedirs { "../ThirdParty" }

    files { "../ThirdParty/stb_image_write.c" }

    filter "configurations:Debug"
        defines { "_DEBUG", "DEBUG" }
        symbols "On"

    filter "configurations:Release"
        defines { "NDEBUG" }
        optimize "On"

project "preprocessor"
    kind "ConsoleApp"
    language "C++"
    location "build/preprocessor"
    targetdir "build/%{cfg.buildcfg}"

    includedirs { "../ThirdParty" }

    files { "preprocessor.cpp" }

    links { "stb_image",
            "stb_image_resize",
            "stb_image_write" }

    filter "system:linux"
        links { "pthread" }

    filter "configurations:Debug"
        defines { "_DEBUG", "DEBUG" }
        symbols "On"

    filter "configurations:Release"
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

    filter { "system:linux", "action:gmake2", "configurations:Debug" }
        buildoptions { "-Wno-unused-but-set-variable",
                       "-Wno-unused-variable",
                       "-Wno-unused-function",
                       "-fno-omit-frame-pointer" }
