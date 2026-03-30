file(GLOB TEST_SOURCES test_*.cpp)

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PRIVATE qe Catch2::Catch2WithMain)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
