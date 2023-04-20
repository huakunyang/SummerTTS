#ifndef _PINYIN_MAP_H_
#define _PINYIN_MAP_H_

#include <vector>
#include <string>
#include <map>

using namespace std;

multimap<string, vector<string> > initPinyin2Phone();
map<string, int32_t> initPhoneIDMap();
map<uint16_t, uint16_t> initNumMap();

#endif

