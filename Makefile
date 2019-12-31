.PHONY: all clean realclean distclean

FIZBO-SRC := \
	board_util.cpp \
	engine.cpp \
	eval.cpp \
	game_util.cpp \
	hash.cpp \
	magic_bb.cpp \
	nn-eval.cpp \
	nn-weights.cpp \
	os-compat.cpp \
	pawn.cpp \
	piece_square.cpp \
	search.cpp \
	tbprobe.cpp \
	threads.cpp

UNIT-TESTS-SRC := unit-tests.cpp

FIZBO-OBJ := $(patsubst %.cpp,.objs/%.o,$(FIZBO-SRC))

UNIT-TESTS-OBJ := $(patsubst %.cpp,.objs/%.o,$(UNIT-TESTS-SRC))

ALL-DEPS := $(patsubst %.cpp,.deps/%.d,$(FIZBO-SRC) $(UNIT-TESTS-SRC))

EXTRAFLAGS := -DNDEBUG
CXXFLAGS := -Wall -Wpedantic -O3 -flto -ggdb -std=c++17 -march=native -pthread $(EXTRAFLAGS)
LIBS :=  -latomic

all: fizbo unit-tests

clean:
	$(RM) -r .deps .objs

realclean: clean
	$(RM) fizbo unit-tests

.objs/%.o: %.cpp Makefile
	mkdir -p .deps .objs
	$(CXX) $(CXXFLAGS) -c -o $@ $< -MMD -MF .deps/$(basename $<).d

include $(wildcard $(ALL-DEPS))

fizbo: $(FIZBO-OBJ)
	$(CXX) $(CXXFLAGS) $(LIBS) -o $@ $(FIZBO-OBJ)

# we'll use -shared here to expose all those inspect symbols
unit-tests: $(UNIT-TESTS-OBJ)
	$(CXX) $(CXXFLAGS) $(LIBS) -o $@ $(UNIT-TESTS-OBJ)
