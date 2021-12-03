set(STORM_HAVE_SPOT OFF)

if(STORM_USE_SPOT_SYSTEM)
    # try to find Spot on the system
    if (NOT "${SPOT_ROOT}" STREQUAL "")
        message(STATUS "Storm - searching for Spot in ${SPOT_ROOT}")
		find_library(SPOT QUIET PATHS ${SPOT_ROOT} NO_DEFAULT_PATH)
    endif()
	if (NOT SPOT_FOUND)
		find_package(SPOT QUIET)
    endif()

    if (NOT SPOT_FOUND)
        set(MANUAL_SPOT_DIR ${SPOT_ROOT})
        ExternalProject_Add(spot
                PREFIX ${SPOT_ROOT}/spot
                SOURCE_DIR ${SPOT_ROOT}/spot_src
                CONFIGURE_COMMAND ${SPOT_ROOT}/spot_src/configure --prefix=${MANUAL_SPOT_DIR}/spot --disable-python
                BUILD_COMMAND make -j${STORM_RESOURCES_BUILD_JOBCOUNT}
                INSTALL_COMMAND make install
                BUILD_BYPRODUCTS ${SPOT_ROOT}/spot/lib/libspot${DYNAMIC_EXT}
                )

        set(SPOT_INCLUDE_DIR "${MANUAL_SPOT_DIR}/spot/include/")
        set(SPOT_DIR "${MANUAL_SPOT_DIR}/spot/")
        set(SPOT_LIBRARIES ${MANUAL_SPOT_DIR}/spot/lib/libspot${DYNAMIC_EXT})
        set(STORM_HAVE_SPOT ON)
        set(SPOT_FOUND ON)
    endif()

    if (SPOT_FOUND)
        message(STATUS "Storm - Using system version of Spot ${SPOT_VERSION} (include: ${SPOT_INCLUDE_DIR}, library: ${SPOT_LIBRARIES}).")
        set(STORM_HAVE_SPOT ON)
    elseif(NOT STORM_USE_SPOT_SHIPPED)
        message (WARNING "Storm - Could not find Spot. Model checking of LTL formulas (beyond PCTL) will not be supported. You may want to set cmake option STORM_USE_SPOT_SHIPPED to install Spot automatically. If you already installed Spot, consider setting cmake option SPOT_ROOT. Unset STORM_USE_SPOT_SYSTEM to silence this warning.")
    endif()
elseif()
endif()

set(STORM_SHIPPED_SPOT OFF)
if(STORM_USE_SPOT_SHIPPED AND NOT STORM_HAVE_SPOT)

    # download and install shipped Spot
    ExternalProject_Add(spot
        URL http://www.lrde.epita.fr/dload/spot/spot-2.9.6.tar.gz # When updating, also change version output below
        DOWNLOAD_NO_PROGRESS TRUE
        DOWNLOAD_DIR ${STORM_3RDPARTY_BINARY_DIR}/spot_src
        SOURCE_DIR ${STORM_3RDPARTY_BINARY_DIR}/spot_src
        PREFIX ${STORM_3RDPARTY_BINARY_DIR}/spot
        CONFIGURE_COMMAND ${STORM_3RDPARTY_BINARY_DIR}/spot_src/configure --prefix=${STORM_3RDPARTY_BINARY_DIR}/spot --disable-python
        BUILD_COMMAND make -j${STORM_RESOURCES_BUILD_JOBCOUNT}
        INSTALL_COMMAND make install
        LOG_CONFIGURE ON
        LOG_BUILD ON
        LOG_INSTALL ON
        BUILD_BYPRODUCTS ${STORM_3RDPARTY_BINARY_DIR}/spot/lib/libspot${DYNAMIC_EXT}
    )
	add_dependencies(resources spot)
    set(SPOT_INCLUDE_DIR "${STORM_3RDPARTY_BINARY_DIR}/spot/include/")
    set(SPOT_DIR "${STORM_3RDPARTY_BINARY_DIR}/spot/")
    set(SPOT_LIBRARIES ${STORM_3RDPARTY_BINARY_DIR}/spot/lib/libspot${DYNAMIC_EXT})
    set(STORM_HAVE_SPOT ON)
    set(STORM_SHIPPED_SPOT ON)

    message(STATUS "Storm - Using shipped version of Spot 2.9.6 (include: ${SPOT_INCLUDE_DIR}, library ${SPOT_LIBRARIES}).")

endif()

if (STORM_HAVE_SPOT)
    include_directories("${SPOT_INCLUDE_DIR}")
    list(APPEND STORM_LINK_LIBRARIES ${SPOT_LIBRARIES})
endif()
