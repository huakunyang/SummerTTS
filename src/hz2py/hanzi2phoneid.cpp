#include "hanzi2phoneid.h"
#include <iostream>
#include <fstream>
#include "Hanz2Piny.h"
#include "pinyinmap.h"
#include "string.h"

hanzi2phoneid::hanzi2phoneid()
{
    pinyMap_ = initPinyin2Phone();
    numMap_ = initNumMap();
    phoneIdMap_ = initPhoneIDMap();    
}

hanzi2phoneid:: ~hanzi2phoneid()
{

}

int32_t * hanzi2phoneid::convert(string line, int32_t & len)
{
    int32_t maxSize = line.size()*4;

    const Hanz2Piny hanz2piny;
    if (hanz2piny.isStartWithBom(line)) 
    {
        line = string(line.cbegin() + 3, line.cend());
    }

    vector<pair<bool, vector<string>>> pinyin_list_list  = hanz2piny.toPinyinFromUtf8(line,numMap_, true, true, "-");

    int32_t * idList = new int32_t[maxSize];
    memset(idList,0,sizeof(int32_t)*maxSize);

    int32_t insertIdx = 0;
    idList[insertIdx++] = 0;

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

    idList[insertIdx++] = 0;
    idList[insertIdx++] = 1;
    len = insertIdx;

    return idList;
    
}
