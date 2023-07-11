// Creates quadrature rule on an edge
//
// Author: Jerome Droniou (jerome.droniou@monash.edu)
//


#include <cmath>
#include <iostream>
#include <limits>
#include <iomanip>

#include "legendregauss.hpp"

using namespace Manicore;

LegendreGauss::LegendreGauss(size_t doe) : _doe(doe),
    _npts(std::ceil((_doe + 1)/ 2.0)),
    _t(0),
    _w(0){
    switch (_npts) {
      case 1:
        sub_rule_01();
        break;
      case 2:
        sub_rule_02();
        break;
      case 3:
        sub_rule_03();
        break;
      case 4:
        sub_rule_04();
        break;
      case 5:
        sub_rule_05();
        break;
      case 6:
        sub_rule_06();
        break;
      case 7:
        sub_rule_07();
        break;
      case 8:
        sub_rule_08();
        break;
      case 9:
        sub_rule_09();
        break;
      case 10:
        sub_rule_10();
        break;
      case 11:
        sub_rule_11();
        break;
      case 12:
        sub_rule_12();
        break;
      case 13:
        sub_rule_13();
        break;
      case 14:
        sub_rule_14();
        break;
      case 15:
        sub_rule_15();
        break;
      case 16:
        sub_rule_16();
        break;
      case 17:
        sub_rule_17();
        break;
      case 18:
        sub_rule_18();
        break;
      case 19:
        sub_rule_19();
        break;
      case 20:
        sub_rule_20();
        break;
      case 21:
        sub_rule_21();
        break;
      default:
        throw "Can't integrate edge to degree ";
        return;
    }
}

LegendreGauss::~LegendreGauss(){
  delete[] _t;
  delete[] _w;
}

void LegendreGauss::sub_rule_01() {
    _w = new double[1];
    _t = new double[1];
    _t[0] = 0.50000;
    _w[0] = 1.0;
}
void LegendreGauss::sub_rule_02() {
    _w = new double[2];
    _t = new double[2];

    _t[0] = 0.5+sqrt(3.0)/6.0;
    _t[1] = 0.5-sqrt(3.0)/6.0;
    _w[0] = 0.5;
    _w[1] = 0.5;
}
void LegendreGauss::sub_rule_03() {
    _w = new double[3];
    _t = new double[3];

    _t[0] = 0.5+ sqrt(15.0)/10.0;
    _t[1] = 0.50000;
    _t[2] = 0.5- sqrt(15.0)/10.0;
    _w[0] = 5.0/18;
    _w[1] = 8.0/18;
    _w[2] = 5.0/18;
}
void LegendreGauss::sub_rule_04() {
    _w = new double[4];
    _t = new double[4];

    _t[0] = 0.5+sqrt(525+70.0*sqrt(30))/70.0;
    _t[1] = 0.5+sqrt(525-70.0*sqrt(30))/70.0;
    _t[2] = 0.5-sqrt(525-70.0*sqrt(30))/70.0;
    _t[3] = 0.5-sqrt(525+70.0*sqrt(30))/70.0;
    _w[0] = (18-sqrt(30.0))/72.0;
    _w[1] = (18+sqrt(30.0))/72.0;
    _w[2] = (18+sqrt(30.0))/72.0;
    _w[3] = (18-sqrt(30.0))/72.0;
}
void LegendreGauss::sub_rule_05() {
    _w = new double[5];
    _t = new double[5];

    _t[0] = 0.5+0.9061798459386640/2.0;
    _t[1] = 0.5+0.5384693101056831/2.0;
    _t[2] = 0.500000;
    _t[3] = 0.5-0.5384693101056831/2.0;
    _t[4] = 0.5-0.9061798459386640/2.0;

    _w[0] = 0.2369268850561891/2.0;
    _w[1] = 0.4786286704993665/2.0;
    _w[2] = 0.5688888888888889/2.0;
    _w[3] = 0.4786286704993665/2.0;
    _w[4] = 0.2369268850561891/2.0;
}
void LegendreGauss::sub_rule_06() {
    _w = new double[6];
    _t = new double[6];

    _t[0] = 0.5+0.9324695142031521/2.0;
    _t[1] = 0.5+0.6612093864662645/2.0;
    _t[2] = 0.5+0.2386191860831969/2.0;
    _t[3] = 0.5-0.2386191860831969/2.0;
    _t[4] = 0.5-0.6612093864662645/2.0;
    _t[5] = 0.5-0.9324695142031521/2.0;

    _w[0] = 0.1713244923791704/2.0;
    _w[1] = 0.3607615730481386/2.0;
    _w[2] = 0.4679139345726910/2.0;
    _w[3] = 0.4679139345726910/2.0;
    _w[4] = 0.3607615730481386/2.0;
    _w[5] = 0.1713244923791704/2.0;
}
void LegendreGauss::sub_rule_07() {
    _w = new double[7];
    _t = new double[7];

    _t[0] = 0.5+0.9491079123427585/2.0;
    _t[1] = 0.5+0.7415311855993945/2.0;
    _t[2] = 0.5+0.4058451513773972/2.0;
    _t[3] = 0.500000;
    _t[4] = 0.5-0.4058451513773972/2.0;
    _t[5] = 0.5-0.7415311855993945/2.0;
    _t[6] = 0.5-0.9491079123427585/2.0;

    _w[0] = 0.1294849661688697/2.0;
    _w[1] = 0.2797053914892766/2.0;
    _w[2] = 0.3818300505051189/2.0;
    _w[3] = 0.4179591836734694/2.0;
    _w[4] = 0.3818300505051189/2.0;
    _w[5] = 0.2797053914892766/2.0;
    _w[6] = 0.1294849661688697/2.0;
}
void LegendreGauss::sub_rule_08() {
    _w = new double[8];
    _t = new double[8];

    _t[0] = 0.5+0.9602898564975363/2.0;
    _t[1] = 0.5+0.7966664774136267/2.0;
    _t[2] = 0.5+0.5255324099163290/2.0;
    _t[3] = 0.5+0.1834346424956498/2.0;
    _t[4] = 0.5-0.1834346424956498/2.0;
    _t[5] = 0.5-0.5255324099163290/2.0;
    _t[6] = 0.5-0.7966664774136267/2.0;
    _t[7] = 0.5-0.9602898564975363/2.0;

    _w[0] = 0.1012285362903763/2.0;
    _w[1] = 0.2223810344533745/2.0;
    _w[2] = 0.3137066458778873/2.0;
    _w[3] = 0.3626837833783620/2.0;
    _w[4] = 0.3626837833783620/2.0;
    _w[5] = 0.3137066458778873/2.0;
    _w[6] = 0.2223810344533745/2.0;
    _w[7] = 0.1012285362903763/2.0;
}
void LegendreGauss::sub_rule_09() {
    _w = new double[9];
    _t = new double[9];

    _t[0] = 0.5+0.9681602395076261/2.0;
    _t[1] = 0.5+0.8360311073266358/2.0;
    _t[2] = 0.5+0.6133714327005904/2.0;
    _t[3] = 0.5+0.3242534234038089/2.0;
    _t[4] = 0.500000;
    _t[5] = 0.5-0.3242534234038089/2.0;
    _t[6] = 0.5-0.6133714327005904/2.0;
    _t[7] = 0.5-0.8360311073266358/2.0;
    _t[8] = 0.5-0.9681602395076261/2.0;

    _w[0] = 0.0812743883615744/2.0;
    _w[1] = 0.1806481606948574/2.0;
    _w[2] = 0.2606106964029354/2.0;
    _w[3] = 0.3123470770400029/2.0;
    _w[4] = 0.3302393550012598/2.0;
    _w[5] = 0.3123470770400029/2.0;
    _w[6] = 0.2606106964029354/2.0;
    _w[7] = 0.1806481606948574/2.0;
    _w[8] = 0.0812743883615744/2.0;
}
void LegendreGauss::sub_rule_10() {
    _w = new double[10];
    _t = new double[10];

    _t[0] = 0.5+0.9739065285171717/2.0;
    _t[1] = 0.5+0.8650633666889845/2.0;
    _t[2] = 0.5+0.6794095682990244/2.0;
    _t[3] = 0.5+0.4333953941292472/2.0;
    _t[4] = 0.5+0.1488743389816312/2.0;
    _t[5] = 0.5-0.1488743389816312/2.0;
    _t[6] = 0.5-0.4333953941292472/2.0;
    _t[7] = 0.5-0.6794095682990244/2.0;
    _t[8] = 0.5-0.8650633666889845/2.0;
    _t[9] = 0.5-0.9739065285171717/2.0;

    _w[0] = 0.0666713443086881/2.0;
    _w[1] = 0.1494513491505806/2.0;
    _w[2] = 0.2190863625159820/2.0;
    _w[3] = 0.2692667193099963/2.0;
    _w[4] = 0.2955242247147529/2.0;
    _w[5] = 0.2955242247147529/2.0;
    _w[6] = 0.2692667193099963/2.0;
    _w[7] = 0.2190863625159820/2.0;
    _w[8] = 0.1494513491505806/2.0;
    _w[9] = 0.0666713443086881/2.0;
}
void LegendreGauss::sub_rule_11() {
    _w = new double[11];
    _t = new double[11];

    _t[0] = 0.5+0.9782286581460570/2.0;
    _t[1] = 0.5+0.8870625997680953/2.0;
    _t[2] = 0.5+0.7301520055740494/2.0;
    _t[3] = 0.5+0.5190961292068118/2.0;
    _t[4] = 0.5+0.2695431559523450/2.0;
    _t[5] = 0.500000000000000000000000;
    _t[6] = 0.5-0.2695431559523450/2.0;
    _t[7] = 0.5-0.5190961292068118/2.0;
    _t[8] = 0.5-0.7301520055740494/2.0;
    _t[9] = 0.5-0.8870625997680953/2.0;
    _t[10] = 0.5-0.9782286581460570/2.0;


    _w[0] = 0.0556685671161737/2.0;
    _w[1] = 0.1255803694649046/2.0;
    _w[2] = 0.1862902109277343/2.0;
    _w[3] = 0.2331937645919905/2.0;
    _w[4] = 0.2628045445102467/2.0;
    _w[5] = 0.2729250867779006/2.0;
    _w[6] = 0.2628045445102467/2.0;
    _w[7] = 0.2331937645919905/2.0;
    _w[8] = 0.1862902109277343/2.0;
    _w[9] = 0.1255803694649046/2.0;
    _w[10] = 0.0556685671161737/2.0;
}
void LegendreGauss::sub_rule_12() {
	_w = new double[12];
	_t = new double[12];

	_t[0] = 0.5-0.9815606342467192/2.0;
	_t[1] = 0.5-0.9041172563704748/2.0;
	_t[2] = 0.5-0.7699026741943047/2.0;
	_t[3] = 0.5-0.5873179542866175/2.0;
	_t[4] = 0.5-0.3678314989981802/2.0;
	_t[5] = 0.5-0.1252334085114689/2.0;
	_t[6] = 0.5+0.1252334085114689/2.0;
	_t[7] = 0.5+0.3678314989981802/2.0;
	_t[8] = 0.5+0.5873179542866175/2.0;
	_t[9] = 0.5+0.7699026741943047/2.0;
	_t[10] = 0.5+0.9041172563704748/2.0;
	_t[11] = 0.5+0.9815606342467192/2.0;

	_w[0] = 0.04717533638651202/2.0;
	_w[1] = 0.10693932599531888/2.0;
	_w[2] = 0.1600783285433461/2.0;
	_w[3] = 0.20316742672306565/2.0;
	_w[4] = 0.23349253653835464/2.0;
	_w[5] = 0.2491470458134027/2.0;
	_w[6] = 0.2491470458134027/2.0;
	_w[7] = 0.23349253653835464/2.0;
	_w[8] = 0.20316742672306565/2.0;
	_w[9] = 0.1600783285433461/2.0;
	_w[10] = 0.10693932599531888/2.0;
	_w[11] = 0.04717533638651202/2.0;
}

void LegendreGauss::sub_rule_13() {
	_w = new double[13];
	_t = new double[13];

	_t[0] = 0.5-0.9841830547185881/2.0;
	_t[1] = 0.5-0.9175983992229779/2.0;
	_t[2] = 0.5-0.8015780907333099/2.0;
	_t[3] = 0.5-0.6423493394403402/2.0;
	_t[4] = 0.5-0.4484927510364468/2.0;
	_t[5] = 0.5-0.23045831595513477/2.0;
	_t[6] = 0.5+0.0/2.0;
	_t[7] = 0.5+0.23045831595513477/2.0;
	_t[8] = 0.5+0.4484927510364468/2.0;
	_t[9] = 0.5+0.6423493394403402/2.0;
	_t[10] = 0.5+0.8015780907333099/2.0;
	_t[11] = 0.5+0.9175983992229779/2.0;
	_t[12] = 0.5+0.9841830547185881/2.0;

	_w[0] = 0.04048400476531588/2.0;
	_w[1] = 0.0921214998377286/2.0;
	_w[2] = 0.13887351021978736/2.0;
	_w[3] = 0.17814598076194552/2.0;
	_w[4] = 0.20781604753688857/2.0;
	_w[5] = 0.22628318026289715/2.0;
	_w[6] = 0.2325515532308739/2.0;
	_w[7] = 0.22628318026289715/2.0;
	_w[8] = 0.20781604753688857/2.0;
	_w[9] = 0.17814598076194552/2.0;
	_w[10] = 0.13887351021978736/2.0;
	_w[11] = 0.0921214998377286/2.0;
	_w[12] = 0.04048400476531588/2.0;
}

void LegendreGauss::sub_rule_14() {
	_w = new double[14];
	_t = new double[14];

	_t[0] = 0.5-0.9862838086968123/2.0;
	_t[1] = 0.5-0.9284348836635735/2.0;
	_t[2] = 0.5-0.827201315069765/2.0;
	_t[3] = 0.5-0.6872929048116855/2.0;
	_t[4] = 0.5-0.5152486363581541/2.0;
	_t[5] = 0.5-0.31911236892788974/2.0;
	_t[6] = 0.5-0.10805494870734367/2.0;
	_t[7] = 0.5+0.10805494870734367/2.0;
	_t[8] = 0.5+0.31911236892788974/2.0;
	_t[9] = 0.5+0.5152486363581541/2.0;
	_t[10] = 0.5+0.6872929048116855/2.0;
	_t[11] = 0.5+0.827201315069765/2.0;
	_t[12] = 0.5+0.9284348836635735/2.0;
	_t[13] = 0.5+0.9862838086968123/2.0;

	_w[0] = 0.035119460331752374/2.0;
	_w[1] = 0.0801580871597603/2.0;
	_w[2] = 0.12151857068790296/2.0;
	_w[3] = 0.1572031671581934/2.0;
	_w[4] = 0.18553839747793763/2.0;
	_w[5] = 0.20519846372129555/2.0;
	_w[6] = 0.21526385346315766/2.0;
	_w[7] = 0.21526385346315766/2.0;
	_w[8] = 0.20519846372129555/2.0;
	_w[9] = 0.18553839747793763/2.0;
	_w[10] = 0.1572031671581934/2.0;
	_w[11] = 0.12151857068790296/2.0;
	_w[12] = 0.0801580871597603/2.0;
	_w[13] = 0.035119460331752374/2.0;
}

void LegendreGauss::sub_rule_15() {
	_w = new double[15];
	_t = new double[15];

	_t[0] = 0.5-0.9879925180204854/2.0;
	_t[1] = 0.5-0.937273392400706/2.0;
	_t[2] = 0.5-0.8482065834104272/2.0;
	_t[3] = 0.5-0.7244177313601701/2.0;
	_t[4] = 0.5-0.5709721726085388/2.0;
	_t[5] = 0.5-0.3941513470775634/2.0;
	_t[6] = 0.5-0.20119409399743451/2.0;
	_t[7] = 0.5+0.0/2.0;
	_t[8] = 0.5+0.20119409399743451/2.0;
	_t[9] = 0.5+0.3941513470775634/2.0;
	_t[10] = 0.5+0.5709721726085388/2.0;
	_t[11] = 0.5+0.7244177313601701/2.0;
	_t[12] = 0.5+0.8482065834104272/2.0;
	_t[13] = 0.5+0.937273392400706/2.0;
	_t[14] = 0.5+0.9879925180204854/2.0;

	_w[0] = 0.030753241996118647/2.0;
	_w[1] = 0.07036604748810807/2.0;
	_w[2] = 0.10715922046717177/2.0;
	_w[3] = 0.1395706779261539/2.0;
	_w[4] = 0.16626920581699378/2.0;
	_w[5] = 0.18616100001556188/2.0;
	_w[6] = 0.19843148532711125/2.0;
	_w[7] = 0.2025782419255609/2.0;
	_w[8] = 0.19843148532711125/2.0;
	_w[9] = 0.18616100001556188/2.0;
	_w[10] = 0.16626920581699378/2.0;
	_w[11] = 0.1395706779261539/2.0;
	_w[12] = 0.10715922046717177/2.0;
	_w[13] = 0.07036604748810807/2.0;
	_w[14] = 0.030753241996118647/2.0;
}

void LegendreGauss::sub_rule_16() {
	_w = new double[16];
	_t = new double[16];

	_t[0] = 0.5-0.9894009349916499/2.0;
	_t[1] = 0.5-0.9445750230732326/2.0;
	_t[2] = 0.5-0.8656312023878318/2.0;
	_t[3] = 0.5-0.755404408355003/2.0;
	_t[4] = 0.5-0.6178762444026438/2.0;
	_t[5] = 0.5-0.45801677765722737/2.0;
	_t[6] = 0.5-0.2816035507792589/2.0;
	_t[7] = 0.5-0.09501250983763745/2.0;
	_t[8] = 0.5+0.09501250983763745/2.0;
	_t[9] = 0.5+0.2816035507792589/2.0;
	_t[10] = 0.5+0.45801677765722737/2.0;
	_t[11] = 0.5+0.6178762444026438/2.0;
	_t[12] = 0.5+0.755404408355003/2.0;
	_t[13] = 0.5+0.8656312023878318/2.0;
	_t[14] = 0.5+0.9445750230732326/2.0;
	_t[15] = 0.5+0.9894009349916499/2.0;

	_w[0] = 0.027152459411754037/2.0;
	_w[1] = 0.062253523938647706/2.0;
	_w[2] = 0.09515851168249259/2.0;
	_w[3] = 0.12462897125553403/2.0;
	_w[4] = 0.14959598881657676/2.0;
	_w[5] = 0.16915651939500262/2.0;
	_w[6] = 0.1826034150449236/2.0;
	_w[7] = 0.18945061045506859/2.0;
	_w[8] = 0.18945061045506859/2.0;
	_w[9] = 0.1826034150449236/2.0;
	_w[10] = 0.16915651939500262/2.0;
	_w[11] = 0.14959598881657676/2.0;
	_w[12] = 0.12462897125553403/2.0;
	_w[13] = 0.09515851168249259/2.0;
	_w[14] = 0.062253523938647706/2.0;
	_w[15] = 0.027152459411754037/2.0;
}

void LegendreGauss::sub_rule_17() {
	_w = new double[17];
	_t = new double[17];

	_t[0] = 0.5-0.9905754753144174/2.0;
	_t[1] = 0.5-0.9506755217687678/2.0;
	_t[2] = 0.5-0.8802391537269859/2.0;
	_t[3] = 0.5-0.7815140038968014/2.0;
	_t[4] = 0.5-0.6576711592166908/2.0;
	_t[5] = 0.5-0.5126905370864769/2.0;
	_t[6] = 0.5-0.3512317634538763/2.0;
	_t[7] = 0.5-0.17848418149584785/2.0;
	_t[8] = 0.5+0.0/2.0;
	_t[9] = 0.5+0.17848418149584785/2.0;
	_t[10] = 0.5+0.3512317634538763/2.0;
	_t[11] = 0.5+0.5126905370864769/2.0;
	_t[12] = 0.5+0.6576711592166908/2.0;
	_t[13] = 0.5+0.7815140038968014/2.0;
	_t[14] = 0.5+0.8802391537269859/2.0;
	_t[15] = 0.5+0.9506755217687678/2.0;
	_t[16] = 0.5+0.9905754753144174/2.0;

	_w[0] = 0.02414830286854952/2.0;
	_w[1] = 0.0554595293739866/2.0;
	_w[2] = 0.08503614831717908/2.0;
	_w[3] = 0.11188384719340365/2.0;
	_w[4] = 0.13513636846852523/2.0;
	_w[5] = 0.15404576107681012/2.0;
	_w[6] = 0.16800410215644995/2.0;
	_w[7] = 0.17656270536699253/2.0;
	_w[8] = 0.17944647035620653/2.0;
	_w[9] = 0.17656270536699253/2.0;
	_w[10] = 0.16800410215644995/2.0;
	_w[11] = 0.15404576107681012/2.0;
	_w[12] = 0.13513636846852523/2.0;
	_w[13] = 0.11188384719340365/2.0;
	_w[14] = 0.08503614831717908/2.0;
	_w[15] = 0.0554595293739866/2.0;
	_w[16] = 0.02414830286854952/2.0;
}

void LegendreGauss::sub_rule_18() {
	_w = new double[18];
	_t = new double[18];

	_t[0] = 0.5-0.9915651684209309/2.0;
	_t[1] = 0.5-0.9558239495713978/2.0;
	_t[2] = 0.5-0.8926024664975557/2.0;
	_t[3] = 0.5-0.8037049589725231/2.0;
	_t[4] = 0.5-0.6916870430603532/2.0;
	_t[5] = 0.5-0.5597708310739475/2.0;
	_t[6] = 0.5-0.41175116146284263/2.0;
	_t[7] = 0.5-0.2518862256915055/2.0;
	_t[8] = 0.5-0.08477501304173529/2.0;
	_t[9] = 0.5+0.08477501304173529/2.0;
	_t[10] = 0.5+0.2518862256915055/2.0;
	_t[11] = 0.5+0.41175116146284263/2.0;
	_t[12] = 0.5+0.5597708310739475/2.0;
	_t[13] = 0.5+0.6916870430603532/2.0;
	_t[14] = 0.5+0.8037049589725231/2.0;
	_t[15] = 0.5+0.8926024664975557/2.0;
	_t[16] = 0.5+0.9558239495713978/2.0;
	_t[17] = 0.5+0.9915651684209309/2.0;

	_w[0] = 0.02161601352648413/2.0;
	_w[1] = 0.04971454889496922/2.0;
	_w[2] = 0.07642573025488925/2.0;
	_w[3] = 0.10094204410628699/2.0;
	_w[4] = 0.12255520671147836/2.0;
	_w[5] = 0.14064291467065063/2.0;
	_w[6] = 0.15468467512626521/2.0;
	_w[7] = 0.16427648374583273/2.0;
	_w[8] = 0.16914238296314363/2.0;
	_w[9] = 0.16914238296314363/2.0;
	_w[10] = 0.16427648374583273/2.0;
	_w[11] = 0.15468467512626521/2.0;
	_w[12] = 0.14064291467065063/2.0;
	_w[13] = 0.12255520671147836/2.0;
	_w[14] = 0.10094204410628699/2.0;
	_w[15] = 0.07642573025488925/2.0;
	_w[16] = 0.04971454889496922/2.0;
	_w[17] = 0.02161601352648413/2.0;
}

void LegendreGauss::sub_rule_19() {
	_w = new double[19];
	_t = new double[19];

	_t[0] = 0.5-0.9924068438435844/2.0;
	_t[1] = 0.5-0.96020815213483/2.0;
	_t[2] = 0.5-0.9031559036148179/2.0;
	_t[3] = 0.5-0.8227146565371428/2.0;
	_t[4] = 0.5-0.7209661773352294/2.0;
	_t[5] = 0.5-0.600545304661681/2.0;
	_t[6] = 0.5-0.46457074137596094/2.0;
	_t[7] = 0.5-0.31656409996362983/2.0;
	_t[8] = 0.5-0.1603586456402254/2.0;
	_t[9] = 0.5+0.0/2.0;
	_t[10] = 0.5+0.1603586456402254/2.0;
	_t[11] = 0.5+0.31656409996362983/2.0;
	_t[12] = 0.5+0.46457074137596094/2.0;
	_t[13] = 0.5+0.600545304661681/2.0;
	_t[14] = 0.5+0.7209661773352294/2.0;
	_t[15] = 0.5+0.8227146565371428/2.0;
	_t[16] = 0.5+0.9031559036148179/2.0;
	_t[17] = 0.5+0.96020815213483/2.0;
	_t[18] = 0.5+0.9924068438435844/2.0;

	_w[0] = 0.01946178822972761/2.0;
	_w[1] = 0.04481422676569981/2.0;
	_w[2] = 0.06904454273764107/2.0;
	_w[3] = 0.09149002162244985/2.0;
	_w[4] = 0.11156664554733375/2.0;
	_w[5] = 0.1287539625393362/2.0;
	_w[6] = 0.14260670217360638/2.0;
	_w[7] = 0.15276604206585945/2.0;
	_w[8] = 0.15896884339395415/2.0;
	_w[9] = 0.16105444984878345/2.0;
	_w[10] = 0.15896884339395415/2.0;
	_w[11] = 0.15276604206585945/2.0;
	_w[12] = 0.14260670217360638/2.0;
	_w[13] = 0.1287539625393362/2.0;
	_w[14] = 0.11156664554733375/2.0;
	_w[15] = 0.09149002162244985/2.0;
	_w[16] = 0.06904454273764107/2.0;
	_w[17] = 0.04481422676569981/2.0;
	_w[18] = 0.01946178822972761/2.0;
}

void LegendreGauss::sub_rule_20() {
	_w = new double[20];
	_t = new double[20];

	_t[0] = 0.5-0.9931285991850949/2.0;
	_t[1] = 0.5-0.9639719272779138/2.0;
	_t[2] = 0.5-0.9122344282513258/2.0;
	_t[3] = 0.5-0.8391169718222188/2.0;
	_t[4] = 0.5-0.7463319064601508/2.0;
	_t[5] = 0.5-0.636053680726515/2.0;
	_t[6] = 0.5-0.5108670019508271/2.0;
	_t[7] = 0.5-0.37370608871541955/2.0;
	_t[8] = 0.5-0.2277858511416451/2.0;
	_t[9] = 0.5-0.07652652113349734/2.0;
	_t[10] = 0.5+0.07652652113349734/2.0;
	_t[11] = 0.5+0.2277858511416451/2.0;
	_t[12] = 0.5+0.37370608871541955/2.0;
	_t[13] = 0.5+0.5108670019508271/2.0;
	_t[14] = 0.5+0.636053680726515/2.0;
	_t[15] = 0.5+0.7463319064601508/2.0;
	_t[16] = 0.5+0.8391169718222188/2.0;
	_t[17] = 0.5+0.9122344282513258/2.0;
	_t[18] = 0.5+0.9639719272779138/2.0;
	_t[19] = 0.5+0.9931285991850949/2.0;

	_w[0] = 0.017614007139153273/2.0;
	_w[1] = 0.04060142980038622/2.0;
	_w[2] = 0.06267204833410944/2.0;
	_w[3] = 0.08327674157670467/2.0;
	_w[4] = 0.10193011981724026/2.0;
	_w[5] = 0.11819453196151825/2.0;
	_w[6] = 0.13168863844917653/2.0;
	_w[7] = 0.14209610931838187/2.0;
	_w[8] = 0.14917298647260366/2.0;
	_w[9] = 0.15275338713072578/2.0;
	_w[10] = 0.15275338713072578/2.0;
	_w[11] = 0.14917298647260366/2.0;
	_w[12] = 0.14209610931838187/2.0;
	_w[13] = 0.13168863844917653/2.0;
	_w[14] = 0.11819453196151825/2.0;
	_w[15] = 0.10193011981724026/2.0;
	_w[16] = 0.08327674157670467/2.0;
	_w[17] = 0.06267204833410944/2.0;
	_w[18] = 0.04060142980038622/2.0;
	_w[19] = 0.017614007139153273/2.0;
}

void LegendreGauss::sub_rule_21() {
	_w = new double[21];
	_t = new double[21];

	_t[0] = 0.5-0.9937521706203895/2.0;
	_t[1] = 0.5-0.9672268385663063/2.0;
	_t[2] = 0.5-0.9200993341504008/2.0;
	_t[3] = 0.5-0.8533633645833173/2.0;
	_t[4] = 0.5-0.7684399634756779/2.0;
	_t[5] = 0.5-0.6671388041974123/2.0;
	_t[6] = 0.5-0.5516188358872198/2.0;
	_t[7] = 0.5-0.4243421202074388/2.0;
	_t[8] = 0.5-0.2880213168024011/2.0;
	_t[9] = 0.5-0.1455618541608951/2.0;
	_t[10] = 0.5+0.0/2.0;
	_t[11] = 0.5+0.1455618541608951/2.0;
	_t[12] = 0.5+0.2880213168024011/2.0;
	_t[13] = 0.5+0.4243421202074388/2.0;
	_t[14] = 0.5+0.5516188358872198/2.0;
	_t[15] = 0.5+0.6671388041974123/2.0;
	_t[16] = 0.5+0.7684399634756779/2.0;
	_t[17] = 0.5+0.8533633645833173/2.0;
	_t[18] = 0.5+0.9200993341504008/2.0;
	_t[19] = 0.5+0.9672268385663063/2.0;
	_t[20] = 0.5+0.9937521706203895/2.0;

	_w[0] = 0.016017228257774137/2.0;
	_w[1] = 0.03695378977085292/2.0;
	_w[2] = 0.057134425426857156/2.0;
	_w[3] = 0.07610011362837935/2.0;
	_w[4] = 0.09344442345603382/2.0;
	_w[5] = 0.10879729916714831/2.0;
	_w[6] = 0.12183141605372842/2.0;
	_w[7] = 0.13226893863333739/2.0;
	_w[8] = 0.13988739479107312/2.0;
	_w[9] = 0.14452440398997007/2.0;
	_w[10] = 0.14608113364969047/2.0;
	_w[11] = 0.14452440398997007/2.0;
	_w[12] = 0.13988739479107312/2.0;
	_w[13] = 0.13226893863333739/2.0;
	_w[14] = 0.12183141605372842/2.0;
	_w[15] = 0.10879729916714831/2.0;
	_w[16] = 0.09344442345603382/2.0;
	_w[17] = 0.07610011362837935/2.0;
	_w[18] = 0.057134425426857156/2.0;
	_w[19] = 0.03695378977085292/2.0;
	_w[20] = 0.016017228257774137/2.0;
}

size_t LegendreGauss::npts() { return _npts; }
double LegendreGauss::wq(size_t i) {
    if (i >= _npts) {
        throw "Trying to access quadrature point that is greater than number computed";
    }
    return _w[i];
}

double LegendreGauss::tq(size_t i) {
    if (i >= _npts) {
        throw "Trying to access quadrature point that is greater than number computed";
    }
    return _t[i];
}
