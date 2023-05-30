// Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <fstream>
#include <iostream>
#include <string>

#include "processor/processor.h"
#include "utils/flags.h"

/*
DEFINE_string(text, "", "input string");
DEFINE_string(file, "", "input file");
DEFINE_string(tagger, "", "tagger fst path");
DEFINE_string(verbalizer, "", "verbalizer fst path");
*/

#include <iostream>
#include <istream>
#include <streambuf>
#include <string>

#include "stdio.h"
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <string.h>
#include <dirent.h>
#include "stdlib.h"
#include <ctime>

struct membuf : std::streambuf
{
    membuf(char* begin, char* end) 
    {
            this->setg(begin, begin, end);
    }
};

char * read_file_buf(char * fileName, int & bufSize)
{

    struct stat st;
    if(-1 == stat(fileName, &st))
    {
        return NULL ;
    }

    FILE *fp = fopen(fileName, "rb");
    if(!fp)
    {
        return NULL;
    }
   

    char *  buf = (char *)malloc(st.st_size);
    fread(buf, st.st_size, 1, fp);
    fclose(fp);
    
    bufSize = st.st_size;
    return buf;
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  int buf1_size = 0;
  int buf2_size = 0;

  char * buf1 = read_file_buf(argv[1], buf1_size);
  char * buf2 = read_file_buf(argv[2], buf2_size);

  membuf sbuf1(buf1, buf1 + buf1_size);
  membuf sbuf2(buf2, buf2 + buf2_size);

  std::istream in1(&sbuf1);
  std::istream in2(&sbuf2);

  wetext::Processor processor(in1, in2);

  

  //if (!FLAGS_text.empty()) 
  {
    std::string tagged_text = processor.tag(argv[3]);
    //std::cout << tagged_text << std::endl;
    std::string normalized_text = processor.verbalize(tagged_text);
    std::cout << normalized_text << std::endl;
  }

  return 0;
}
