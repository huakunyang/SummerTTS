#include "EnglishText2Id.h"
#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <Eigen/Dense>
#include <unordered_map>
#include <vector>
#include "nn_sigmoid.h"
#include "nn_tanh.h"
#include <vector>
#include <algorithm>

using Eigen::MatrixXf;
using Eigen::Map;

extern const char * Alphabet[];
extern const wchar_t * Ipa[];

#define IPA_NUM (125000)

typedef struct
{
    //vector<string> ipaSymbolsVec_;
    wstring ipaSymbols_;
    unordered_map<string, wstring> ipaTable_;
    unordered_map<string, int> char2Id_;
    unordered_map<int,string> id2Phone_;
    unordered_map<string,wstring> phone2ipa_;
    MatrixXf enc_emb_;
    MatrixXf enc_w_ih_;
    MatrixXf enc_w_hh_;
    MatrixXf enc_b_ih_;
    MatrixXf enc_b_hh_;
    MatrixXf dec_emb_;
    MatrixXf dec_w_ih_;
    MatrixXf dec_w_hh_;
    MatrixXf dec_b_ih_;
    MatrixXf dec_b_hh_;
    MatrixXf fc_w_;
    MatrixXf fc_b_;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}ENG_TEXT_2_ID_DATA_t;

void deleteNumFromStr(string & str)
{
    string::iterator it = str.begin();
    while (it != str.end()) 
    {
        if (((*it >= '1') && (*it <= '9'))||(*it == '0')) 
        {
            it = str.erase(it);
        } 
        else 
        {
            it++;
        }
    }
}

EnglishText2Id::EnglishText2Id(float * modelData, int32_t & offset)
{
    ENG_TEXT_2_ID_DATA_t * engText2idData = new ENG_TEXT_2_ID_DATA_t();

    engText2idData->ipaSymbols_= L"_;:,.!?¡¿—…\"«»“” ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";

    for(int32_t i = 0; i<IPA_NUM; i++)
    {
        engText2idData->ipaTable_.insert(make_pair<string, wstring>(Alphabet[i],Ipa[i]));
    }

    int32_t curOffset = offset;
    int32_t x = (int32_t)modelData[curOffset++];
    int32_t y = (int32_t)modelData[curOffset++];
    engText2idData->enc_emb_ = Map<MatrixXf>(modelData+curOffset, x, y);
    curOffset = curOffset + x*y;

    x = (int32_t)modelData[curOffset++];
    y = (int32_t)modelData[curOffset++];
    engText2idData->enc_w_ih_ = Map<MatrixXf>(modelData+curOffset, x, y);
    curOffset = curOffset + x*y;
    
    x = (int32_t)modelData[curOffset++];
    y = (int32_t)modelData[curOffset++];
    engText2idData->enc_w_hh_ = Map<MatrixXf>(modelData+curOffset, x, y);
    curOffset = curOffset + x*y;

    x = (int32_t)modelData[curOffset++];
    engText2idData->enc_b_ih_ = Map<MatrixXf>(modelData+curOffset, 1, x);
    curOffset = curOffset + x;
    
    x = (int32_t)modelData[curOffset++];
    engText2idData->enc_b_hh_ = Map<MatrixXf>(modelData+curOffset, 1, x);
    curOffset = curOffset + x;

    x = (int32_t)modelData[curOffset++];
    y = (int32_t)modelData[curOffset++];
    engText2idData->dec_emb_ = Map<MatrixXf>(modelData+curOffset, x, y);
    curOffset = curOffset + x*y;

    x = (int32_t)modelData[curOffset++];
    y = (int32_t)modelData[curOffset++];
    engText2idData->dec_w_ih_ = Map<MatrixXf>(modelData+curOffset, x, y);
    curOffset = curOffset + x*y;

    x = (int32_t)modelData[curOffset++];
    y = (int32_t)modelData[curOffset++];
    engText2idData->dec_w_hh_ = Map<MatrixXf>(modelData+curOffset, x, y);
    curOffset = curOffset + x*y;

    x = (int32_t)modelData[curOffset++];
    engText2idData->dec_b_ih_ = Map<MatrixXf>(modelData+curOffset, 1, x);
    curOffset = curOffset + x;
    
    x = (int32_t)modelData[curOffset++];
    engText2idData->dec_b_hh_ = Map<MatrixXf>(modelData+curOffset, 1, x);
    curOffset = curOffset + x;
    
    x = (int32_t)modelData[curOffset++];
    y = (int32_t)modelData[curOffset++];
    engText2idData->fc_w_ = Map<MatrixXf>(modelData+curOffset, x, y);
    curOffset = curOffset + x*y;

    x = (int32_t)modelData[curOffset++];
    engText2idData->fc_b_ = Map<MatrixXf>(modelData+curOffset, 1, x);
    curOffset = curOffset + x;

    offset = curOffset;

    engText2idData->char2Id_.insert(make_pair<string, int>("<pad>",0));
    engText2idData->char2Id_.insert(make_pair<string, int>("<unk>",1));
    engText2idData->char2Id_.insert(make_pair<string, int>("</s>",2));
    engText2idData->char2Id_.insert(make_pair<string, int>("a",3));
    engText2idData->char2Id_.insert(make_pair<string, int>("b",4));
    engText2idData->char2Id_.insert(make_pair<string, int>("c",5));
    engText2idData->char2Id_.insert(make_pair<string, int>("d",6));
    engText2idData->char2Id_.insert(make_pair<string, int>("e",7));
    engText2idData->char2Id_.insert(make_pair<string, int>("f",8));
    engText2idData->char2Id_.insert(make_pair<string, int>("g",9));
    engText2idData->char2Id_.insert(make_pair<string, int>("h",10));
    engText2idData->char2Id_.insert(make_pair<string, int>("i",11));
    engText2idData->char2Id_.insert(make_pair<string, int>("j",12));
    engText2idData->char2Id_.insert(make_pair<string, int>("k",13));
    engText2idData->char2Id_.insert(make_pair<string, int>("l",14));
    engText2idData->char2Id_.insert(make_pair<string, int>("m",15));
    engText2idData->char2Id_.insert(make_pair<string, int>("n",16));
    engText2idData->char2Id_.insert(make_pair<string, int>("o",17));
    engText2idData->char2Id_.insert(make_pair<string, int>("p",18));
    engText2idData->char2Id_.insert(make_pair<string, int>("q",19));
    engText2idData->char2Id_.insert(make_pair<string, int>("r",20));
    engText2idData->char2Id_.insert(make_pair<string, int>("s",21));
    engText2idData->char2Id_.insert(make_pair<string, int>("t",22));
    engText2idData->char2Id_.insert(make_pair<string, int>("u",23));
    engText2idData->char2Id_.insert(make_pair<string, int>("v",24));
    engText2idData->char2Id_.insert(make_pair<string, int>("w",25));
    engText2idData->char2Id_.insert(make_pair<string, int>("x",26));
    engText2idData->char2Id_.insert(make_pair<string, int>("y",27));
    engText2idData->char2Id_.insert(make_pair<string, int>("z",28));

    engText2idData->id2Phone_.insert(make_pair<int,string>(0,"<pad>"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(1,"<unk>"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(2,"<s>"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(3,"</s>"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(4,"AA0"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(5,"AA1"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(6,"AA2"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(7,"AE0"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(8,"AE1"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(9,"AE2"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(10,"AH0"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(11,"AH1"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(12,"AH2"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(13,"AO0"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(14,"AO1"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(15,"AO2"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(16,"AW0"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(17,"AW1"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(18,"AW2"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(19,"AY0"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(20,"AY1"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(21,"AY2"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(22,"B"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(23,"CH"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(24,"D"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(25,"DH"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(26,"EH0"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(27,"EH1"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(28,"EH2"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(29,"ER0"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(30,"ER1"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(31,"ER2"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(32,"EY0"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(33,"EY1"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(34,"EY2"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(35,"F"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(36,"G"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(37,"HH"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(38,"IH0"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(39,"IH1"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(40,"IH2"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(41,"IY0"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(42,"IY1"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(43,"IY2"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(44,"JH"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(45,"K"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(46,"L"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(47,"M"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(48,"N"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(49,"NG"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(50,"OW0"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(51,"OW1"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(52,"OW2"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(53,"OY0"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(54,"OY1"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(55,"OY2"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(56,"P"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(57,"R"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(58,"S"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(59,"SH"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(60,"T"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(61,"TH"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(62,"UH0"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(63,"UH1"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(64,"UH2"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(65,"UW"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(66,"UW0"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(67,"UW1"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(68,"UW2"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(69,"V"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(70,"W"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(71,"Y"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(72,"Z"));
    engText2idData->id2Phone_.insert(make_pair<int,string>(73,"ZH"));
    
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("a",L"ə"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("ey",L"eɪ"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("aa",L"ɑ"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("ae",L"æ"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("ah",L"ə"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("ao",L"ɔ"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("aw",L"aʊ"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("ay",L"aɪ"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("ch",L"ʧ"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("dh",L"ð"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("eh",L"ɛ"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("er",L"ər"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("hh",L"h"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("ih",L"ɪ"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("jh",L"ʤ"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("ng",L"ŋ"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("ow",L"oʊ"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("oy",L"ɔɪ"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("sh",L"ʃ"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("th",L"θ"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("uh",L"ʊ"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("uw",L"u"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("zh",L"ʒ"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("iy",L"i"));
    engText2idData->phone2ipa_.insert(make_pair<string, wstring>("y",L"j"));
    
    priv_ = (void *)engText2idData;

}

EnglishText2Id::~EnglishText2Id()
{
    
}

MatrixXf gru_cell(const MatrixXf & x,    const MatrixXf & h, 
                  const MatrixXf & w_ih, const MatrixXf & w_hh, 
                  const MatrixXf & b_ih, const MatrixXf & b_hh)
{
    MatrixXf rzn_ih = x*(w_ih.transpose()) + b_ih;
    MatrixXf rzn_hh = h*(w_hh.transpose()) + b_hh;

    MatrixXf rz_ih =  rzn_ih.block(0,0,rzn_ih.rows(),(int32_t)(rzn_ih.cols()*2/3));
    MatrixXf n_ih  =  rzn_ih.block(0,(int32_t)(rzn_ih.cols()*2/3),rzn_ih.rows(),rzn_ih.cols() - (int32_t)(rzn_ih.cols()*2/3));

    MatrixXf rz_hh = rzn_hh.block(0,0,rzn_hh.rows(),(int32_t)(rzn_hh.cols()*2/3));
    MatrixXf n_hh  = rzn_hh.block(0,(int32_t)(rzn_hh.cols()*2/3), rzn_hh.rows(), rzn_hh.cols() - (int32_t)(rzn_hh.cols()*2/3));

    MatrixXf rz = nn_sigmoid(rz_ih + rz_hh);

    MatrixXf r = rz.block(0,0,rz.rows(), rz.cols()/2);
    MatrixXf z = rz.block(0,rz.cols()/2,rz.rows(),rz.cols()/2);
   
    MatrixXf n = nn_tanh(n_ih.array() + r.array() * n_hh.array());

    MatrixXf hh = ((z.array()*(-1)+1)*n.array()+(z.array() * h.array())).matrix();

    return hh;
}
                
MatrixXf gru(const MatrixXf & x,    int32_t steps, 
             const MatrixXf & w_ih, const MatrixXf & w_hh, 
             const MatrixXf & b_ih, const MatrixXf & b_hh,
             const MatrixXf & h0)
{
    MatrixXf output = MatrixXf::Zero(x.rows(),x.cols());

    MatrixXf hh = h0;
    for(int32_t i = 0; i< steps; i++)
    {
        hh = gru_cell(x.row(i), hh, w_ih, w_hh, b_ih, b_hh);
        output.row(i) = hh;
    }

    return output;
}

string replaceNum(const string & str)
{
    string ret;
    int32_t insertPos = 0;
    for(int32_t i = 0; i<str.length(); i++)
    {
        switch(str[i])
        {
            case '0':
                ret.insert(insertPos," zero ");
                insertPos += 6;
                break;

            case '1':
                ret.insert(insertPos," one ");
                insertPos += 5;
                break;

            case '2':
                ret.insert(insertPos," two ");
                insertPos += 5;
                break;

            case '3':
                ret.insert(insertPos," three ");
                insertPos += 7;
                break;

            case '4':
                ret.insert(insertPos," four ");
                insertPos += 6;
                break;

            case '5':
                ret.insert(insertPos," five ");
                insertPos += 6;
                break;

            case '6':
                ret.insert(insertPos," six ");
                insertPos += 5;
                break;

            case '7':
                ret.insert(insertPos," seven ");
                insertPos += 7;
                break;

            case '8':
                ret.insert(insertPos," eight ");
                insertPos += 7;
                break;

            case '9':
                ret.insert(insertPos," nine ");
                insertPos += 6;
                break;

            default:
                ret.push_back(str[i]);
                insertPos++;


        }
    }

    return ret;
}

typedef struct
{
    int32_t needProcess_;
    string oriString_;
}PROCESS_UNIT_t;


vector<PROCESS_UNIT_t> preProcess(const string & str)
{
    string symbols ="_;:,.!?¡¿—…\"«»“”";
    istringstream in(str);

    vector<PROCESS_UNIT_t> retVec;

    string word;
    while (in >> word) 
    {

        auto c = word[word.length()-1];

        bool found = false;
        for(int32_t i=0; i<symbols.length(); i++)
        {
            if(symbols[i] == c)
            {
                found = true;
                break;
            }
        }

        if(found == false)
        {
            PROCESS_UNIT_t processUnit;
            processUnit.needProcess_ = 1;
            processUnit.oriString_ = word;
            retVec.push_back(processUnit);
        }
        else
        {
            string removedLastStr = word;
            removedLastStr.erase(word.length()-1,1);

            PROCESS_UNIT_t processUnit;
            processUnit.needProcess_ = 1;
            processUnit.oriString_ = removedLastStr;
            retVec.push_back(processUnit);

            PROCESS_UNIT_t processUnit2;
            processUnit2.needProcess_ = 0;
            processUnit2.oriString_.push_back(c);
            retVec.push_back(processUnit2);
        }

    }

    return retVec;
}

std::wstring s2ws(const std::string& str) 
{
    if (str.empty()) 
    {
        return L"";
    }

    unsigned len = str.size() + 1;
    setlocale(LC_CTYPE, "en_US.UTF-8");
    wchar_t *p = new wchar_t[len];
    mbstowcs(p, str.c_str(), len);
    std::wstring w_str(p);
    delete[] p;
    return w_str;
}


vector<int> EnglishText2Id::getIPAId(const string & strEng)
{
    ENG_TEXT_2_ID_DATA_t * engText2idData = (ENG_TEXT_2_ID_DATA_t *)priv_;

    string strEngNumReplaced = replaceNum(strEng);

    vector<int> ret;
    istringstream in(strEngNumReplaced);

    vector<PROCESS_UNIT_t> processVec =  preProcess(strEngNumReplaced);

    vector<wstring> vecIPAs;
    for(int32_t idx = 0; idx<processVec.size(); idx++)
    {
        PROCESS_UNIT_t pu = processVec[idx];

        if(pu.needProcess_ == 1)
        {
            string word = pu.oriString_;

            transform(word.begin(),word.end(),word.begin(),::tolower);
            auto it = engText2idData->ipaTable_.find(word);
            if(it != engText2idData->ipaTable_.end())
            {
                wstring ipaStr = it->second;
                vecIPAs.push_back(ipaStr);
            }
            else
            {
                int32_t wordSize = word.length();

                if(wordSize <4)
                {
                    for(int32_t i = 0; i<wordSize; i++)
                    {
                        wstring oneChar;
                        oneChar.push_back(word[i]);
                        vecIPAs.push_back(oneChar);
                    }
                }
                else
                {
                    MatrixXf enc = MatrixXf::Zero(wordSize+1, engText2idData->enc_emb_.cols());
                    for(int32_t lIdx = 0; lIdx < wordSize; lIdx++)
                    {
                        string letterStr;
                        letterStr.insert(0,1,word[lIdx]);

                        auto it = engText2idData->char2Id_.find(letterStr);
                        if(it != engText2idData->char2Id_.end())
                        {
                            enc.row(lIdx) =  engText2idData->enc_emb_.row(it->second);
                        }
                        else
                        {
                            enc.row(lIdx) =  engText2idData->enc_emb_.row(1);
                        }
                    }
                    enc.row(wordSize) = engText2idData->enc_emb_.row(2);

                    MatrixXf h0 = MatrixXf::Zero(1,engText2idData->enc_w_hh_.cols());
                    MatrixXf gru_out = gru(enc, wordSize+1, 
                                        engText2idData->enc_w_ih_, engText2idData->enc_w_hh_,
                                        engText2idData->enc_b_ih_, engText2idData->enc_b_hh_,h0);
                                    
                    MatrixXf last_hidden = gru_out.row(gru_out.rows()-1);
                    MatrixXf dec = engText2idData->dec_emb_.row(2);
                    MatrixXf h = last_hidden;

                    std::vector<int> preds;

                    for(int32_t i=0; i<20; i++)
                    {
                        h = gru_cell(dec, h, engText2idData->dec_w_ih_, 
                                    engText2idData->dec_w_hh_, 
                                    engText2idData->dec_b_ih_, 
                                    engText2idData->dec_b_hh_);

                        MatrixXf logits = (h*(engText2idData->fc_w_).transpose()) + engText2idData->fc_b_;

                        MatrixXf::Index maxRow, maxCol;
                        logits.maxCoeff(&maxRow,&maxCol);


                        if(maxCol == 3)
                            break;

                        preds.push_back(maxCol);
                        dec = engText2idData->dec_emb_.row(maxCol);

                    }
                    
                    wstring predPhones;
                    for(int32_t i = 0; i<preds.size(); i++)
                    {
                        string phone = engText2idData->id2Phone_[preds[i]];
                        deleteNumFromStr(phone);
                        transform(phone.begin(),phone.end(),phone.begin(),::tolower);
                
                        auto it = engText2idData->phone2ipa_.find(phone);
                        if(it != engText2idData->phone2ipa_.end())
                        {
                            predPhones.append(engText2idData->phone2ipa_[phone]);
                    
                        }
                        else
                        {
                            predPhones.append(s2ws(phone));
                        }

                    }
                    vecIPAs.push_back(predPhones);
                }
            }
        }
        else
        {
            vecIPAs.push_back(s2ws(pu.oriString_));            
        }
        
    }

    for(int32_t i = 0; i<vecIPAs.size(); i++)
    {
        wstring onePhone = vecIPAs[i];
        int32_t onePhoneSize = vecIPAs[i].length();

        for(int32_t j = 0; j<onePhoneSize ; j++)
        {
            bool foundPhone = false;
            for(int32_t k = 0; k< engText2idData->ipaSymbols_.size(); k++)
            {
                if(engText2idData->ipaSymbols_[k] == onePhone[j])
                {
                    ret.push_back(0);
                    ret.push_back(k);
                    foundPhone = true;
                    break;
                }
            }

            if(foundPhone == false)
            {
                ret.push_back(16);
            }
            //ret.push_back(0);
        }
        ret.push_back(0);
        ret.push_back(16);
    }

    return ret;
}

