# Copyright (C) 2018 Bho Matthiesen
# 
# This program is used in the article:
# 
# Bho Matthiesen, Alessio Zappone, Eduard A. Jorswieck, and Merouane Debbah,
# "Deep Learning for Optimal Energy-Efficient Power Control in Wireless
# Interference Networks," submitted to IEEE Journal on Selected Areas in
# Communication.
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

## default targets

.PHONY:
all: wsee_lambert

.PHONY:
clean:
	rm -f *.o *.a
	rm -rf .d

## rules

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

