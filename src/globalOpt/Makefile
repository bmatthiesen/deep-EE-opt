# Copyright (C) 2018-2019 Bho Matthiesen, Karl-Ludwig Besser
# 
# This program is used in the article:
# 
# Bho Matthiesen, Alessio Zappone, Karl-L. Besser, Eduard A. Jorswieck, and
# Merouane Debbah, "A Globally Optimal Energy-Efficient Power Control Framework
# and its Efficient Implementation in Wireless Interference Networks,"
# submitted to IEEE Transactions on Signal Processing
# 
# License:
# This program is licensed under the GPLv2 license. If you in any way use this
# code for research that results in publications, please cite our original
# article listed above.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

## config

CXXFLAGS=-std=c++17
CPPFLAGS=-march=native -O3 -Wall -DNDEBUG -fpic -g -Wall -Wno-format -Wno-unused-parameters -fopenmp
#CPPFLAGS=-g -O0 -Wall -DDEBUG -fpic
LDFLAGS=-g -march=native -m64
LDLIBS=-lstdc++ -lm

# intel mkl
LDLIBS +=  -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_rt -lpthread -lm -ldl
CPPFLAGS +=  -m64 -I${MKLROOT}/include

# lambertW
CPPFLAGS += -Ilambert_w/include

# path to anaconda python3 binary
PY = ${HOME}/anaconda3/bin/python3

## default targets

.PHONY:
all: wseePy.cpython-36m-x86_64-linux-gnu.so

.PHONY:
clean:
	rm -f *.o *.a
	rm -rf .d

## rules
wseePy.cpython-36m-x86_64-linux-gnu.so: setup.py wseePy.pyx util.o wsee_lambert.h BRB.h
	$(PY) $< build
	ln -sf build/lib.linux-x86_64-3.6/$@

wseePy.pyx: wseePy4.pyx wseePy7.pyx
	cat $^ > $@
	@rm $^

wseePy%.pyx: wseePy.pyx.m4
	m4 -P --define=DIM=$(*F) $< > $@

test_lambert: util.o

### automatic dependency tracking

SRCS = $(wildcard *.cpp)

DEPDIR := .d
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td

COMPILE.c = $(CC) $(DEPFLAGS) $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c
COMPILE.cc = $(CXX) $(DEPFLAGS) $(CXXFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c
POSTCOMPILE = @mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d && touch $@

%.o : %.c
%.o : %.c $(DEPDIR)/%.d
	$(COMPILE.c) $(OUTPUT_OPTION) $<
	$(POSTCOMPILE)

%.o : %.cc
%.o : %.cc $(DEPDIR)/%.d
	$(COMPILE.cc) $(OUTPUT_OPTION) $<
	$(POSTCOMPILE)

%.o : %.cxx
%.o : %.cxx $(DEPDIR)/%.d
	$(COMPILE.cc) $(OUTPUT_OPTION) $<
	$(POSTCOMPILE)

%.o : %.cpp
%.o : %.cpp $(DEPDIR)/%.d
	$(COMPILE.cc) $(OUTPUT_OPTION) $<
	$(POSTCOMPILE)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d


include $(wildcard $(patsubst %,$(DEPDIR)/%.d,$(basename $(SRCS))))

