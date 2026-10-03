# Машинный отбор

| элемент | выбран | CLAP | PQ | забраковано |
|---|---|---|---|---|
| winter_outside | dsp21 | 0.385 | 5.89 | 3 из 6 |
| hut_air | 39b9acs14 | 0.223 | 4.95 | 3 из 4 |
| fire_burning | d0ee6as12 | 0.550 | 6.93 | 1 из 4 |
| door_open | ca0bf1s18 | 0.210 | 5.39 | 6 из 8 |
| door_close | 861802s13 | 0.045 | 5.90 | 7 из 8 |
| floor_walk | 504d73s12 ⚠ все плохи, взят наименее плохой | 0.115 | 6.45 | 8 из 8 |
| table_thud | 415873s15 | 0.292 | 5.96 | 6 из 8 |
| cough | 464948s15 ⚠ все плохи, взят наименее плохой | 0.057 | 4.25 | 8 из 8 |
| fire_catching | 9f2679s14 | 0.504 | 6.26 | 7 из 8 |

## winter_outside — цель: fierce winter wind and blizzard outside | strong winter blizzard wind blowing
- ✔ `winter_outside__dsp21` CLAP 0.385, PQ 5.89, RMS 0.065 — годен. AST: Eruption 0.12, White noise 0.07, Waves, surf 0.06
- · `winter_outside__dsp22` CLAP 0.373, PQ 5.79, RMS 0.050 — годен. AST: Eruption 0.13, White noise 0.07, Field recording 0.07
- · `winter_outside__f14446s13` CLAP 0.373, PQ 4.63, RMS 0.101 — годен. AST: Thunder 0.54, Thunderstorm 0.46, Rain 0.13
- ✘ `winter_outside__f14446s11` CLAP 0.361, PQ 4.16, RMS 0.171 — AST: engine 0.23; AST: mic_wind 0.41. AST: Wind noise (microphone) 0.41, Wind 0.37, Vehicle 0.23
- ✘ `winter_outside__f14446s12` CLAP 0.327, PQ 4.66, RMS 0.217 — AST: engine 0.39; AST: mic_wind 0.44. AST: Wind noise (microphone) 0.44, Vehicle 0.39, Wind 0.31
- ✘ `winter_outside__f14446s14` CLAP 0.315, PQ 4.41, RMS 0.263 — AST: engine 0.30; AST: mic_wind 0.34. AST: Wind noise (microphone) 0.34, Boat, Water vehicle 0.33, Vehicle 0.30

## hut_air — цель: still cold air in a wooden room | the quiet interior of a room
- ✘ `hut_air__39b9acs13` CLAP 0.233, PQ 5.31, RMS 0.019 — CLAP: «rustling of dry hay» 0.52. AST: Typewriter 0.20, Typing 0.19, Computer keyboard 0.13
- ✔ `hut_air__39b9acs14` CLAP 0.223, PQ 4.95, RMS 0.031 — годен. AST: Patter 0.27, Printer 0.06, Inside, small room 0.05
- ✘ `hut_air__39b9acs12` CLAP 0.123, PQ 5.10, RMS 0.033 — CLAP: «rain falling» 0.86. AST: Hiss 0.20, Steam 0.17, Vacuum cleaner 0.08
- ✘ `hut_air__39b9acs11` CLAP 0.120, PQ 5.22, RMS 0.018 — CLAP: «rain falling» 0.98. AST: Boiling 0.29, Liquid 0.11, Inside, small room 0.04

## fire_burning — цель: wood fire crackling in a stove | a wood fire crackling
- ✔ `fire_burning__d0ee6as12` CLAP 0.550, PQ 6.93, RMS 0.029 — годен. AST: Crumpling, crinkling 0.44, Inside, small room 0.20, Crushing 0.14
- · `fire_burning__d0ee6as13` CLAP 0.525, PQ 7.08, RMS 0.019 — годен. AST: Crushing 0.32, Crumpling, crinkling 0.20, Crack 0.15
- · `fire_burning__d0ee6as14` CLAP 0.456, PQ 6.34, RMS 0.057 — годен. AST: Chop 0.56, Crushing 0.10, Crackle 0.05
- ✘ `fire_burning__d0ee6as11` CLAP 0.408, PQ 5.93, RMS 0.021 — AST: rain 0.16. AST: Fire 0.19, Crackle 0.17, Rain 0.16

## door_open — цель: heavy wooden door creaking open; ждём: Door, Creak, Squeak
- ✔ `door_open__ca0bf1s18` CLAP 0.210, PQ 5.39, RMS 0.057 — годен. AST: Door 0.36, Slam 0.21, Sliding door 0.17
- ✘ `door_open__ca0bf1s12` CLAP 0.146, PQ 6.17, RMS 0.046 — AST: слышно Wood 0.26, а Door только 0.06. AST: Wood 0.26, Dishes, pots, and pans 0.10, Door 0.06
- · `door_open__ca0bf1s15` CLAP 0.095, PQ 5.52, RMS 0.079 — годен. AST: Door 0.17, Music 0.09, Slam 0.04
- ✘ `door_open__ca0bf1s14` CLAP 0.013, PQ 6.15, RMS 0.106 — AST: слышно Music 0.29, а Door только 0.01. AST: Music 0.29, Speech 0.20, Coin (dropping) 0.10
- ✘ `door_open__ca0bf1s17` CLAP -0.004, PQ 6.48, RMS 0.163 — AST: music 0.47; AST: слышно Music 0.47, а Door только 0.01; CLAP: «rustling of dry hay» 0.76. AST: Music 0.47, Musical instrument 0.15, Percussion 0.06
- ✘ `door_open__ca0bf1s11` CLAP 0.008, PQ 5.52, RMS 0.391 — AST: слышно Scrape 0.13, а Door только 0.03; CLAP: «a car engine revving» 0.80. AST: Scrape 0.13, Music 0.10, Wood 0.09
- ✘ `door_open__ca0bf1s13` CLAP 0.002, PQ 4.52, RMS 0.135 — AST: music 0.54; AST: слышно Music 0.54, а Door только 0.02. AST: Music 0.54, Drum 0.45, Musical instrument 0.36
- ✘ `door_open__ca0bf1s16` CLAP -0.044, PQ 5.07, RMS 0.267 — AST: music 0.49; AST: слышно Music 0.49, а Door только 0.04; CLAP: «a car engine revving» 0.98. AST: Music 0.49, Heart sounds, heartbeat 0.24, Throbbing 0.21

## door_close — цель: heavy wooden door creaking shut; ждём: Door, Creak, Thump, thud
- ✘ `door_close__861802s12` CLAP 0.114, PQ 5.93, RMS 0.164 — AST: слышно Heart sounds, heartbeat 0.19, а Door только 0.02. AST: Heart sounds, heartbeat 0.19, Heart murmur 0.14, Throbbing 0.08
- ✘ `door_close__861802s17` CLAP 0.063, PQ 5.92, RMS 0.231 — AST: слышно Sliding door 0.04, а Door только 0.02. AST: Sliding door 0.04, Music 0.04, Breaking 0.03
- ✔ `door_close__861802s13` CLAP 0.045, PQ 5.90, RMS 0.344 — годен. AST: Door 0.10, Breaking 0.07, Music 0.07
- ✘ `door_close__861802s11` CLAP 0.008, PQ 5.06, RMS 0.244 — AST: music 0.32; AST: слышно Music 0.32, а Creak только 0.01; CLAP: «a car engine revving» 0.79. AST: Music 0.32, Sound effect 0.10, Coin (dropping) 0.05
- ✘ `door_close__861802s14` CLAP -0.040, PQ 6.43, RMS 0.098 — AST: music 0.35; AST: слышно Music 0.35, а Door только 0.14; CLAP: «music playing» 0.98. AST: Music 0.35, Door 0.14, Sliding door 0.04
- ✘ `door_close__861802s18` CLAP -0.068, PQ 5.65, RMS 0.371 — AST: music 0.34; CLAP: «hail hitting a roof» 0.62. AST: Music 0.34, Door 0.22, Slam 0.10
- ✘ `door_close__861802s15` CLAP -0.061, PQ 4.70, RMS 0.055 — AST: music 0.39; AST: слышно Music 0.39, а Door только 0.01; CLAP: «hail hitting a roof» 0.81. AST: Music 0.39, Sound effect 0.06, Television 0.05
- ✘ `door_close__861802s16` CLAP -0.139, PQ 4.71, RMS 0.289 — AST: слышно Music 0.21, а Door только 0.09; CLAP: «hail hitting a roof» 0.79. AST: Music 0.21, Door 0.09, Sliding door 0.06

## floor_walk — цель: heavy boots walking on wooden floorboards; ждём: Walk, footsteps, Creak
- ✘ `floor_walk__504d73s13` CLAP 0.221, PQ 7.15, RMS 0.092 — AST: speech 0.33; AST: слышно Thunk 0.42, а Walk, footsteps только 0.01. AST: Thunk 0.42, Door 0.42, Speech 0.33
- ✔ `floor_walk__504d73s12` CLAP 0.115, PQ 6.45, RMS 0.235 — AST: слышно Knock 0.11, а Walk, footsteps только 0.00. AST: Knock 0.11, Chop 0.07, Whack, thwack 0.06
- ✘ `floor_walk__504d73s18` CLAP 0.128, PQ 5.68, RMS 0.158 — AST: слышно Music 0.29, а Walk, footsteps только 0.00. AST: Music 0.29, Wood block 0.19, Percussion 0.17
- ✘ `floor_walk__504d73s15` CLAP 0.111, PQ 5.78, RMS 0.191 — AST: слышно Heart sounds, heartbeat 0.14, а Walk, footsteps только 0.01. AST: Heart sounds, heartbeat 0.14, Music 0.09, Heart murmur 0.08
- ✘ `floor_walk__504d73s17` CLAP 0.059, PQ 6.48, RMS 0.207 — AST: слышно Typewriter 0.18, а Walk, footsteps только 0.00. AST: Typewriter 0.18, Computer keyboard 0.09, Typing 0.05
- ✘ `floor_walk__504d73s16` CLAP 0.062, PQ 5.81, RMS 0.103 — AST: слышно Music 0.12, а Walk, footsteps только 0.01. AST: Music 0.12, Speech 0.12, Clip-clop 0.04
- ✘ `floor_walk__504d73s11` CLAP 0.040, PQ 6.22, RMS 0.398 — AST: music 0.44; AST: слышно Music 0.44, а Walk, footsteps только 0.00. AST: Music 0.44, Typewriter 0.26, Drum 0.04
- ✘ `floor_walk__504d73s14` CLAP 0.005, PQ 6.83, RMS 0.094 — AST: music 0.64; AST: слышно Music 0.64, а Walk, footsteps только 0.00. AST: Music 0.64, Wood block 0.05, Percussion 0.05

## table_thud — цель: heavy stone pot thudding on wood; ждём: Thump, thud, Thunk, Knock
- ✘ `table_thud__415873s14` CLAP 0.325, PQ 5.94, RMS 0.186 — AST: слышно Bang 0.22, а Thunk только 0.07. AST: Bang 0.22, Burst, pop 0.09, Slam 0.08
- ✘ `table_thud__415873s17` CLAP 0.292, PQ 6.80, RMS 0.084 — AST: слышно Scrape 0.10, а Thunk только 0.02. AST: Sound effect 0.15, Scrape 0.10, Door 0.05
- ✔ `table_thud__415873s15` CLAP 0.292, PQ 5.96, RMS 0.137 — годен. AST: Door 0.14, Knock 0.12, Tap 0.06
- ✘ `table_thud__415873s13` CLAP 0.272, PQ 6.08, RMS 0.124 — AST: слышно Music 0.18, а Thunk только 0.03. AST: Music 0.18, Sound effect 0.08, Glass 0.06
- ✘ `table_thud__415873s11` CLAP 0.246, PQ 5.42, RMS 0.208 — AST: слышно Speech 0.08, а Thunk только 0.01. AST: Speech 0.08, Music 0.08, Breaking 0.07
- ✘ `table_thud__415873s18` CLAP 0.192, PQ 6.07, RMS 0.111 — AST: слышно Music 0.15, а Knock только 0.02. AST: Music 0.15, Hammer 0.08, Wood block 0.06
- ✘ `table_thud__415873s16` CLAP 0.103, PQ 5.16, RMS 0.391 — AST: слышно Speech 0.24, а Thunk только 0.03. AST: Speech 0.24, Music 0.13, Bang 0.13
- · `table_thud__415873s12` CLAP 0.102, PQ 4.57, RMS 0.366 — годен. AST: Knock 0.40, Heart sounds, heartbeat 0.14, Throbbing 0.06

## cough — цель: deep wet rasping human cough; ждём: Cough, Throat clearing
- ✘ `cough__464948s11` CLAP 0.065, PQ 5.11, RMS 0.453 — AST: music 0.68; AST: слышно Music 0.68, а Throat clearing только 0.00; CLAP: «a car engine revving» 0.98. AST: Music 0.68, Sound effect 0.10, Oink 0.10
- ✘ `cough__464948s12` CLAP 0.091, PQ 3.72, RMS 0.622 — AST: music 0.67; AST: слышно Music 0.67, а Throat clearing только 0.00. AST: Music 0.67, Animal 0.08, Scrape 0.06
- ✔ `cough__464948s15` CLAP 0.057, PQ 4.25, RMS 0.502 — AST: слышно Scrape 0.46, а Throat clearing только 0.00. AST: Scrape 0.46, Speech 0.28, Music 0.23
- ✘ `cough__464948s16` CLAP 0.047, PQ 4.15, RMS 0.472 — AST: music 0.59; AST: слышно Music 0.59, а Throat clearing только 0.00. AST: Music 0.59, Scrape 0.13, Sound effect 0.09
- ✘ `cough__464948s17` CLAP 0.018, PQ 5.32, RMS 0.222 — AST: music 0.65; AST: слышно Music 0.65, а Throat clearing только 0.01. AST: Music 0.65, Burping, eructation 0.28, Sound effect 0.08
- ✘ `cough__464948s14` CLAP 0.034, PQ 3.76, RMS 0.524 — AST: слышно Music 0.30, а Throat clearing только 0.00; CLAP: «a car engine revving» 1.00. AST: Music 0.30, Gurgling 0.07, Stomach rumble 0.05
- ✘ `cough__464948s13` CLAP -0.002, PQ 4.83, RMS 0.623 — AST: music 0.65; AST: слышно Scrape 0.68, а Throat clearing только 0.00; CLAP: «a car engine revving» 0.79. AST: Scrape 0.68, Music 0.65, Sound effect 0.28
- ✘ `cough__464948s18` CLAP -0.010, PQ 4.10, RMS 0.611 — AST: music 0.51; AST: слышно Music 0.51, а Throat clearing только 0.00; CLAP: «a car engine revving» 0.93. AST: Music 0.51, Speech 0.09, Sound effect 0.07

## fire_catching — цель: dry wood fire catching and growing; ждём: Fire, Crackle
- ✘ `fire_catching__9f2679s12` CLAP 0.543, PQ 6.60, RMS 0.011 — AST: слышно Scrape 0.07, а Fire только 0.04. AST: Scrape 0.07, Fire 0.04, Crackle 0.04
- ✘ `fire_catching__9f2679s15` CLAP 0.520, PQ 6.26, RMS 0.011 — AST: слышно Arrow 0.19, а Fire только 0.07. AST: Arrow 0.19, Skateboard 0.08, Fire 0.07
- ✘ `fire_catching__9f2679s11` CLAP 0.508, PQ 6.64, RMS 0.005 — AST: слышно Crumpling, crinkling 0.23, а Crackle только 0.04. AST: Crumpling, crinkling 0.23, Inside, small room 0.22, Tearing 0.12
- ✘ `fire_catching__9f2679s18` CLAP 0.506, PQ 6.71, RMS 0.007 — AST: слышно Crushing 0.17, а Crackle только 0.04. AST: Crushing 0.17, Tearing 0.13, Scrape 0.09
- ✔ `fire_catching__9f2679s14` CLAP 0.504, PQ 6.26, RMS 0.019 — годен. AST: Crackle 0.13, Fire 0.10, Bicycle 0.10
- ✘ `fire_catching__9f2679s16` CLAP 0.493, PQ 6.53, RMS 0.009 — AST: слышно Crumpling, crinkling 0.50, а Crackle только 0.04. AST: Crumpling, crinkling 0.50, Inside, small room 0.28, Tearing 0.25
- ✘ `fire_catching__9f2679s17` CLAP 0.437, PQ 6.70, RMS 0.004 — AST: слышно Tearing 0.21, а Crackle только 0.03. AST: Tearing 0.21, Inside, small room 0.16, Crumpling, crinkling 0.14
- ✘ `fire_catching__9f2679s13` CLAP 0.404, PQ 6.77, RMS 0.010 — AST: слышно Crushing 0.19, а Crackle только 0.08. AST: Crushing 0.19, Crumpling, crinkling 0.17, Crackle 0.08
