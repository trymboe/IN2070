def uloba(timer, overtid, kveld, helg, helligdag):
    timelønn = 186
    k_tillegg = 68
    h_tillegg = 65
    hd_tillegg = timelønn*1.3
    o_tillegg = timelønn*2
    return 1.04*(timelønn*timer + overtid*o_tillegg + kveld*k_tillegg + helg*h_tillegg + helligdag*hd_tillegg), timer

def støttekontakt(timer, utlegg, km):
    return timer*155+utlegg+km*5.05, timer

def dalveien(vakter, helg):
    timer = vakter[0]*(7.5+1/6) + vakter[1]*7.5 + vakter[2]*7 + vakter[3]*(7+1/6) + vakter[4]*7 + vakter[5]*6.5
    kveld = vakter[3]*(5+1/6) + vakter[4]*5 + vakter[5]*4.5
    return timer*169.5 + kveld*56 + helg*53, timer

lønnS, timerS = støttekontakt(37.25, 165, 150)
lønnU, timerU = uloba(38.75, 0, 28, 12, 0)
lønnD, timerD = dalveien([1.5, 3, 2, 2, 0, 3], 7.5+1/6)
tot = lønnS + lønnU + lønnD

timer = timerS + timerU + timerD

print("støttekontakt:", lønnS)
print("Uloba:", lønnU)
print("Dalveien:", lønnD)
print("Timer:", timer)
print("timelønn:", lønnU/timerU)

print(tot)


print(dalveien([1, 5.3, 2, 5, 3, 3], 43.5))