#ifndef  _H_ENGLISH_TEXT_2_ID_H_
#define  _H_ENGLISH_TEXT_2_ID_H_

#include <string>
#include <vector>
#include <stdint.h>

using namespace std;

class EnglishText2Id
{
public:
    EnglishText2Id(float * modelData, int32_t & offset);
    ~EnglishText2Id();
    vector<int> getIPAId(const string & strEng);

private:
    void * priv_;

};

#endif
