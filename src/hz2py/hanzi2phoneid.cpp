#include "hanzi2phoneid.h"
#include "./utf8/utf8.h"
#include <iostream>
#include <fstream>
#include "Hanz2Piny.h"
#include "pinyinmap.h"
#include "string.h"
#include <iostream>
#include <string>

using namespace std;

hanzi2phoneid::hanzi2phoneid(std::istream & streamWords, std::istream & streamPinyin)
{
    pinyMap_ = initPinyin2Phone();
    numMap_ = initNumMap();
    phoneIdMap_ = initPhoneIDMap();    

    initMultiPhoneMap(streamWords, streamPinyin);
}

hanzi2phoneid:: ~hanzi2phoneid()
{

}

void hanzi2phoneid::initMultiPhoneMap(std::istream & streamWords, std::istream & streamPinyin )
{
    multimap<string, vector<string>> multiPhoneMap;

    std::string wordLine;
    while(getline(streamWords, wordLine))
    {
        std::string multiPhoneLine;
        getline(streamPinyin, multiPhoneLine);
        
        vector<string> phonesList;
        int32_t splitIdx = 0; 
        int32_t splitIdxPre = splitIdx;
        splitIdx = multiPhoneLine.find(",",splitIdxPre);

        while(splitIdx >  0)
        {
            string phone = multiPhoneLine.substr(splitIdxPre, splitIdx - splitIdxPre);

            phonesList.push_back(phone);

            splitIdxPre = splitIdx+1;
            splitIdx = multiPhoneLine.find(",",splitIdxPre);
        }
        
        multiPhoneMap_.insert(pair<string, vector<string>>(wordLine, phonesList));
    }
}

vector<string> hanzi2phoneid::searchForMultiPhone(const string & word)
{
    auto iter = multiPhoneMap_.find(word);
    vector<string> phones;

    if (iter != multiPhoneMap_.end())
    {
        phones = iter->second;
    }

    return phones;
}

int32_t * hanzi2phoneid::convert(string line, int32_t & len, const vector<string> & jiebaWordsIn)
{
    int32_t maxSize = line.size()*4;

    const Hanz2Piny hanz2piny;
    if (hanz2piny.isStartWithBom(line)) 
    {
        line = string(line.cbegin() + 3, line.cend());
    }

    vector<string> jiebaWords = jiebaWordsIn;

    for(int32_t ii= 0; ii<jiebaWords.size(); ii++)
    {
        string s = jiebaWords[ii];
        
        bool bDummyReplace = false;
        string dummyStr="";
        for (auto iter = s.cbegin(); iter != s.cend(); NULL) 
        {
            auto iter_old = iter;
            utf8::next(iter, s.cend());
            unsigned short unicode;
            utf8::utf8to16(iter_old, iter, &unicode);

            if(!hanz2piny.isHanziUnicode(unicode))
            {
                bDummyReplace = true;
                dummyStr.append("AAA");
                //break;
            }
        }
        
        if(bDummyReplace == true)
        {
            jiebaWords[ii] = dummyStr;
        }
    }

    vector<pair<bool, vector<string>>> pinyin_list_list  = hanz2piny.toPinyinFromUtf8(line,numMap_, true, true, "-");

    int32_t * idList = new int32_t[maxSize];
    memset(idList,0,sizeof(int32_t)*maxSize);

    int32_t insertIdx = 0;
    idList[insertIdx++] = 0;

    int32_t charIdx = 0;
    for (const auto& e : pinyin_list_list)
    {
        const bool ok = e.first;
        auto pinyin_list = e.second;
        auto pinyin = pinyin_list[0];

        int32_t cStrIdx = pinyin.find(":");
        if(cStrIdx < pinyin.size())
        {
            pinyin.erase(cStrIdx,1);
        }

        if(pinyin=="-")
        {
            idList[insertIdx++] = phoneIdMap_["sil"];
            idList[insertIdx++] = phoneIdMap_["sil"];
        }
        else
        {
            int32_t cStrIdx = pinyin.find(",");
            if(cStrIdx < pinyin.size())
            {
                int32_t jiebaSearchIdx = 0;
                for(int32_t wordIdx = 0; wordIdx < jiebaWords.size(); wordIdx++)
                {
                    int32_t jiebaSearchPre = jiebaSearchIdx;
                    jiebaSearchIdx = jiebaSearchIdx + (jiebaWords[wordIdx].length()/3);

                    if(jiebaSearchIdx > charIdx)
                    {
                        vector<string> wordPhones = searchForMultiPhone(jiebaWords[wordIdx]);
                        
                        if(wordPhones.size() == 0)
                        {
                            pinyin = pinyin.substr(0,cStrIdx);
                            string strTone = pinyin.substr(pinyin.size()-1,1);
                            string pinyin_notone = pinyin.substr(0,pinyin.size()-1);

                            auto iter = pinyMap_.find(pinyin_notone);
                            if (iter != pinyMap_.end())
                            {
                                vector<string> phones = iter->second;
                                string a1 = phones[0];
                                string a2 = phones[1]+strTone;

                                idList[insertIdx++] = phoneIdMap_[a1];
                                idList[insertIdx++] = phoneIdMap_[a2];
                                idList[insertIdx++] = phoneIdMap_["#0"];
                            }
                        }
                        else
                        {
                            int32_t letterIdx = charIdx - (jiebaSearchIdx - (jiebaWords[wordIdx].length()/3));
                            
                            pinyin = wordPhones[letterIdx];
                            string strTone = pinyin.substr(pinyin.size()-1,1);
                            string pinyin_notone = pinyin.substr(0,pinyin.size()-1);

                            auto iter = pinyMap_.find(pinyin_notone);
                            if (iter != pinyMap_.end())
                            {
                                vector<string> phones = iter->second;
                                string a1 = phones[0];
                                string a2 = phones[1]+strTone;

                                idList[insertIdx++] = phoneIdMap_[a1];
                                idList[insertIdx++] = phoneIdMap_[a2];
                                idList[insertIdx++] = phoneIdMap_["#0"];
                            }

                        }
                        break;
                    }
                }
                    
            }
            else
            {
                string strTone = pinyin.substr(pinyin.size()-1,1);
                string pinyin_notone = pinyin.substr(0,pinyin.size()-1);

                auto iter = pinyMap_.find(pinyin_notone);
                if (iter != pinyMap_.end())
                {
                    vector<string> phones = iter->second;
                    string a1 = phones[0];
                    string a2 = phones[1]+strTone;

                    idList[insertIdx++] = phoneIdMap_[a1];
                    idList[insertIdx++] = phoneIdMap_[a2];
                    idList[insertIdx++] = phoneIdMap_["#0"];
                }
            }
        }

        charIdx = charIdx + 1;
    }

    idList[insertIdx++] = 0;
    idList[insertIdx++] = 1;
    len = insertIdx;

    return idList;
    
}
