#ifndef _HANZI2PHONEID_H_
#define _HANZI2PHONEID_H_

#include <string>
#include <map>
#include <vector>

using namespace std;

class hanzi2phoneid
{
public:
    hanzi2phoneid(std::istream & streamWords, std::istream & streamPinyin);
    ~hanzi2phoneid();
    int32_t * convert(string line, int32_t & len, const vector<string> & jiebaWords);
    
private:
    vector<string> searchForMultiPhone(const string & word);
    void initMultiPhoneMap(std::istream & streamWords, std::istream & streamPinyin);
    multimap<string, vector<string> >  pinyMap_;
    map<uint16_t, uint16_t>  numMap_;
    map<string, int32_t> phoneIdMap_;

    multimap<string, vector<string> >  multiPhoneMap_;

};

#endif
