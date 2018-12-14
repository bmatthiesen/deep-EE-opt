"""Hata-COST231 Channel Model

Copyright (C) 2018 Bho Matthiesen

This program is used in the article:

Bho Matthiesen, Alessio Zappone, Eduard A. Jorswieck, and Merouane Debbah,
"Deep Learning for Optimal Energy-Efficient Power Control in Wireless
Interference Networks," submitted to IEEE Journal on Selected Areas in
Communication.

License:
This program is licensed under the GPLv2 license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

import numpy as np

def urban(d, fc, Hb, Hm, metropolitan = True):
    """Urban Scenario from Hata-COST231 Model[1]

    Parameter:
        fc: Carrier Frequency in MHz (1500 - 2000)
        Hb: BS Height in m (30 - 200)
        Hm: Mobile Height in m (1 - 10)
         d: Distance in km (1 - 20)

        metropolitan: True for Metropolitan Center, False for medium sized city
                      and suburban areas

    Returns Power Path Loss in dB

    Suitable Parameters for Urban (from [2, Section 6.3]):
        fc = 1900
        Hb = 30
        Hm = 1.5
         d <= .5

    References:
        [1] 3GPP, "Digital cellular telecommunications systems (Phase 2+); Radio
            network planning aspects." 3GPP ETSI TR 43.030, V9.0.0, Feb. 2010.

        [2] T. L. Marzetta, E. G. Larsson, H. Yang, and H. Q. Ngo,
            "Fundamentals of Massive MIMO," Cambridge University Press, 2016.
    """
    log = np.log10

    assert(fc >= 1500 and fc <= 2000)
    assert(Hb >= 30 and Hb <= 200)
    assert(Hm >= 1 and Hm <= 10)
    assert(np.all(d <= 20)) # note nomial applicable range is d>=1km

    if metropolitan:
        Cm = 3
    else:
        Cm = 0

    a = (1.1*log(fc) - 0.7)*Hm - (1.56*log(fc) - 0.8)

    return 46.3 + 33.9*log(fc) - 13.82*log(Hb) - a + (44.9 - 6.55*log(Hb))*log(d) + Cm

def suburban(d, fc, Hb, Hm):
    """Suburban Scenario from Hata-COST231 Model[1]

    Parameter:
        fc: Carrier Frequency in MHz (1500 - 2000)
        Hb: BS Height in m (30 - 200)
        Hm: Mobile Height in m (1 - 10)
         d: Distance in km (1 - 20)

    Suitable Parameters for Suburban (from [2, Section 6.3]):
        fc = 1900
        Hb = 30
        Hm = 1.5
         d <= 2

    Returns Power Path Loss in dB

    References:
        [1] 3GPP, "Digital cellular telecommunications systems (Phase 2+); Radio
            network planning aspects." 3GPP ETSI TR 43.030, V9.0.0, Feb. 2010.

        [2] T. L. Marzetta, E. G. Larsson, H. Yang, and H. Q. Ngo,
            "Fundamentals of Massive MIMO," Cambridge University Press, 2016.
    """
    log = np.log10

    Lu = urban(d, fc, Hb, Hm, False)
    return Lu - 2*(log(fc/28))**2 - 5.4

def ruralQuasiOpen(d, fc, Hb, Hm):
    """Rural (Quasi-open) Scenario from Hata-COST231 Model[1]

    Parameter:
        fc: Carrier Frequency in MHz (1500 - 2000)
        Hb: BS Height in m (30 - 200)
        Hm: Mobile Height in m (1 - 10)
         d: Distance in km (1 - 20)

    Returns Power Path Loss in dB

    References:
        [1] 3GPP, "Digital cellular telecommunications systems (Phase 2+); Radio
            network planning aspects." 3GPP ETSI TR 43.030, V9.0.0, Feb. 2010.
    """
    log = np.log10

    Lu = urban(d, fc, Hb, Hm, False)
    return Lu - 4.78*(log(fc))**2 + 18.33*log(fc) - 35.94

def ruralOpen(d, fc, Hb, Hm):
    """Rural (Open Area) Scenario from Hata-COST231 Model[1]

    Parameter:
        fc: Carrier Frequency in MHz (1500 - 2000)
        Hb: BS Height in m (30 - 200)
        Hm: Mobile Height in m (1 - 10)
         d: Distance in km (1 - 20)

    Returns Power Path Loss in dB

    References:
        [1] 3GPP, "Digital cellular telecommunications systems (Phase 2+); Radio
            network planning aspects." 3GPP ETSI TR 43.030, V9.0.0, Feb. 2010.
    """
    log = np.log10

    Lu = urban(d, fc, Hb, Hm, False)
    return Lu - 4.78*(log(fc))**2 + 18.33*log(fc) - 40.94
