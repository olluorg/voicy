# Машинный отбор

| элемент | выбран | CLAP | PQ | забраковано |
|---|---|---|---|---|
| winter_outside | dsp21 | 0.385 | 5.89 | 4 из 6 |
| hut_air | 39b9acs11 | 0.083 | 7.04 | 3 из 4 |
| fire_burning | d0ee6as13 | 0.512 | 5.99 | 0 из 4 |
| door_open | ab0c00s14 | 0.150 | 5.97 | 5 из 8 |
| door_close | 541fdas13 | 0.286 | 7.28 | 2 из 8 |
| floor_walk | 04d27cs11 ⚠ все плохи, взят наименее плохой | 0.400 | 6.77 | 8 из 8 |
| table_thud | acbbees15 | 0.269 | 7.00 | 0 из 8 |
| cough | fc91c4s11 | 0.389 | 6.45 | 0 из 4 |
| fire_catching | 564219s11 ⚠ все плохи, взят наименее плохой | 0.404 | 7.27 | 8 из 8 |

## winter_outside — цель: fierce winter wind and blizzard outside | strong winter blizzard wind blowing
- ✔ `winter_outside__dsp21` CLAP 0.385, PQ 5.89, RMS 0.065 — годен. AST: Eruption 0.12, White noise 0.07, Waves, surf 0.06
- · `winter_outside__dsp22` CLAP 0.373, PQ 5.79, RMS 0.050 — годен. AST: Eruption 0.13, White noise 0.07, Field recording 0.07
- ✘ `winter_outside__f14446s12` CLAP 0.328, PQ 5.74, RMS 0.085 — AST: engine 0.46. AST: Vehicle 0.46, Field recording 0.43, Aircraft 0.36
- ✘ `winter_outside__f14446s14` CLAP 0.330, PQ 5.63, RMS 0.135 — AST: engine 0.35. AST: Field recording 0.49, Vehicle 0.35, Aircraft 0.34
- ✘ `winter_outside__f14446s11` CLAP 0.310, PQ 5.42, RMS 0.098 — AST: engine 0.32. AST: Field recording 0.50, Vehicle 0.32, Eruption 0.17
- ✘ `winter_outside__f14446s13` CLAP 0.262, PQ 5.82, RMS 0.142 — AST: engine 0.31. AST: Field recording 0.35, Vehicle 0.31, Fixed-wing aircraft, airplane 0.23

## hut_air — цель: still cold air in a wooden room | the quiet interior of a room
- ✔ `hut_air__39b9acs11` CLAP 0.083, PQ 7.04, RMS 0.032 — годен. AST: Coin (dropping) 0.12, Creak 0.06, Thunk 0.05
- ✘ `hut_air__39b9acs13` CLAP 0.068, PQ 6.80, RMS 0.012 — CLAP: «rustling of dry hay» 0.94. AST: Wood 0.23, Thunk 0.17, Drawer open or close 0.17
- ✘ `hut_air__39b9acs14` CLAP 0.038, PQ 6.68, RMS 0.027 — CLAP: «rain falling» 0.99. AST: Mechanisms 0.13, Chop 0.08, Creak 0.03
- ✘ `hut_air__39b9acs12` CLAP -0.033, PQ 6.87, RMS 0.029 — CLAP: «rustling of dry hay» 1.00. AST: Thunk 0.18, Music 0.06, Crushing 0.06

## fire_burning — цель: wood fire crackling in a stove | a wood fire crackling
- ✔ `fire_burning__d0ee6as13` CLAP 0.512, PQ 5.99, RMS 0.012 — годен. AST: Fire 0.10, Hands 0.03, Tap 0.03
- · `fire_burning__d0ee6as11` CLAP 0.514, PQ 5.81, RMS 0.016 — годен. AST: Throbbing 0.16, Heart sounds, heartbeat 0.13, Hum 0.09
- · `fire_burning__d0ee6as12` CLAP 0.502, PQ 5.90, RMS 0.028 — годен. AST: Fire 0.21, Throbbing 0.09, Hum 0.04
- · `fire_burning__d0ee6as14` CLAP 0.480, PQ 5.61, RMS 0.029 — годен. AST: Fire 0.17, Throbbing 0.11, Heart sounds, heartbeat 0.06

## door_open — цель: heavy wooden door creaking open; ждём: Door, Creak, Squeak
- ✔ `door_open__ab0c00s14` CLAP 0.150, PQ 5.97, RMS 0.099 — годен. AST: Knock 0.54, Thunk 0.42, Door 0.35
- · `door_open__ab0c00s15` CLAP 0.137, PQ 6.46, RMS 0.061 — годен. AST: Door 0.29, Knock 0.19, Slam 0.08
- ✘ `door_open__ab0c00s12` CLAP 0.126, PQ 6.87, RMS 0.074 — AST: слышно Knock 0.89, а Door только 0.08. AST: Knock 0.89, Door 0.08, Music 0.08
- · `door_open__ab0c00s11` CLAP 0.121, PQ 6.44, RMS 0.072 — годен. AST: Door 0.46, Knock 0.30, Thunk 0.13
- ✘ `door_open__ab0c00s16` CLAP 0.102, PQ 6.59, RMS 0.052 — CLAP: «rustling of dry hay» 0.73. AST: Door 0.32, Knock 0.26, Thunk 0.12
- ✘ `door_open__ab0c00s13` CLAP 0.100, PQ 6.63, RMS 0.077 — AST: слышно Knock 0.43, а Door только 0.04. AST: Knock 0.43, Thunk 0.16, Door 0.04
- ✘ `door_open__ab0c00s17` CLAP 0.089, PQ 6.90, RMS 0.067 — AST: слышно Knock 0.60, а Door только 0.32. AST: Knock 0.60, Door 0.32, Thunk 0.19
- ✘ `door_open__ab0c00s18` CLAP 0.073, PQ 6.55, RMS 0.057 — CLAP: «rustling of dry hay» 0.91. AST: Knock 0.23, Door 0.20, Thunk 0.19

## door_close — цель: heavy wooden door creaking shut; ждём: Door, Creak, Thump, thud
- ✔ `door_close__541fdas13` CLAP 0.286, PQ 7.28, RMS 0.082 — годен. AST: Door 0.66, Knock 0.32, Slam 0.17
- · `door_close__541fdas15` CLAP 0.283, PQ 6.62, RMS 0.089 — годен. AST: Door 0.67, Slam 0.19, Sliding door 0.18
- · `door_close__541fdas12` CLAP 0.245, PQ 6.78, RMS 0.085 — годен. AST: Door 0.49, Slam 0.22, Sliding door 0.21
- · `door_close__541fdas18` CLAP 0.248, PQ 6.40, RMS 0.072 — годен. AST: Door 0.43, Knock 0.12, Sliding door 0.09
- ✘ `door_close__541fdas17` CLAP 0.221, PQ 6.68, RMS 0.100 — AST: слышно Knock 0.58, а Door только 0.19. AST: Knock 0.58, Thunk 0.22, Door 0.19
- · `door_close__541fdas16` CLAP 0.222, PQ 6.56, RMS 0.091 — годен. AST: Door 0.55, Thunk 0.24, Sliding door 0.16
- ✘ `door_close__541fdas14` CLAP 0.204, PQ 6.66, RMS 0.070 — AST: слышно Knock 0.70, а Door только 0.19. AST: Knock 0.70, Thunk 0.20, Door 0.19
- · `door_close__541fdas11` CLAP 0.181, PQ 6.39, RMS 0.075 — годен. AST: Sound effect 0.26, Door 0.13, Sliding door 0.09

## floor_walk — цель: footsteps on wooden floorboards; ждём: Walk, footsteps, Creak
- ✘ `floor_walk__04d27cs12` CLAP 0.402, PQ 6.93, RMS 0.035 — AST: слышно Chopping (food) 0.10, а Walk, footsteps только 0.03. AST: Chopping (food) 0.10, Knock 0.09, Walk, footsteps 0.03
- ✘ `floor_walk__04d27cs16` CLAP 0.406, PQ 6.64, RMS 0.059 — AST: слышно Knock 0.37, а Walk, footsteps только 0.05. AST: Knock 0.37, Tap 0.08, Shuffling cards 0.06
- ✘ `floor_walk__04d27cs18` CLAP 0.398, PQ 6.98, RMS 0.043 — AST: слышно Knock 0.43, а Walk, footsteps только 0.02. AST: Knock 0.43, Chopping (food) 0.15, Door 0.04
- ✔ `floor_walk__04d27cs11` CLAP 0.400, PQ 6.77, RMS 0.042 — AST: слышно Knock 0.40, а Walk, footsteps только 0.01. AST: Knock 0.40, Shuffling cards 0.10, Chopping (food) 0.07
- ✘ `floor_walk__04d27cs13` CLAP 0.398, PQ 6.54, RMS 0.038 — AST: слышно Knock 0.58, а Walk, footsteps только 0.05. AST: Knock 0.58, Door 0.05, Walk, footsteps 0.05
- ✘ `floor_walk__04d27cs17` CLAP 0.380, PQ 6.94, RMS 0.040 — AST: слышно Knock 0.36, а Walk, footsteps только 0.02. AST: Knock 0.36, Door 0.12, Tap 0.05
- ✘ `floor_walk__04d27cs14` CLAP 0.388, PQ 6.55, RMS 0.039 — AST: слышно Knock 0.46, а Walk, footsteps только 0.04. AST: Knock 0.46, Thunk 0.10, Walk, footsteps 0.04
- ✘ `floor_walk__04d27cs15` CLAP 0.360, PQ 6.78, RMS 0.064 — AST: слышно Knock 0.76, а Walk, footsteps только 0.01. AST: Knock 0.76, Door 0.07, Wood 0.04

## table_thud — цель: heavy stone pot thudding on wood; ждём: Thump, thud, Thunk, Knock
- ✔ `table_thud__acbbees15` CLAP 0.269, PQ 7.00, RMS 0.061 — годен. AST: Knock 0.45, Door 0.03, Tap 0.03
- · `table_thud__acbbees14` CLAP 0.264, PQ 6.82, RMS 0.066 — годен. AST: Knock 0.72, Door 0.25, Slam 0.10
- · `table_thud__acbbees13` CLAP 0.266, PQ 6.66, RMS 0.065 — годен. AST: Knock 0.74, Door 0.12, Thunk 0.05
- · `table_thud__acbbees12` CLAP 0.245, PQ 7.12, RMS 0.066 — годен. AST: Knock 0.54, Thunk 0.02, Sound effect 0.02
- · `table_thud__acbbees16` CLAP 0.252, PQ 6.66, RMS 0.058 — годен. AST: Knock 0.76, Door 0.04, Thunk 0.02
- · `table_thud__acbbees11` CLAP 0.236, PQ 6.62, RMS 0.072 — годен. AST: Knock 0.14, Door 0.11, Slam 0.06
- · `table_thud__acbbees17` CLAP 0.217, PQ 7.12, RMS 0.062 — годен. AST: Door 0.35, Knock 0.26, Thunk 0.10
- · `table_thud__acbbees18` CLAP 0.213, PQ 6.81, RMS 0.076 — годен. AST: Knock 0.65, Door 0.09, Tap 0.03

## cough — цель: deep rasping cough; ждём: Cough, Throat clearing
- ✔ `cough__fc91c4s11` CLAP 0.389, PQ 6.45, RMS 0.067 — годен. AST: Throat clearing 0.55, Cough 0.06, Sound effect 0.05
- · `cough__fc91c4s14` CLAP 0.367, PQ 6.80, RMS 0.082 — годен. AST: Throat clearing 0.83, Cough 0.12, Speech 0.04
- · `cough__fc91c4s12` CLAP 0.357, PQ 6.31, RMS 0.078 — годен. AST: Throat clearing 0.65, Speech 0.12, Burping, eructation 0.08
- · `cough__fc91c4s13` CLAP 0.244, PQ 6.67, RMS 0.086 — годен. AST: Throat clearing 0.25, Speech 0.06, Slap, smack 0.04

## fire_catching — цель: dry wood tinder catching fire and crackling; ждём: Fire, Crackle
- ✘ `fire_catching__564219s16` CLAP 0.441, PQ 7.61, RMS 0.013 — AST: слышно Camera 0.24, а Crackle только 0.01. AST: Camera 0.24, Single-lens reflex camera 0.08, Cap gun 0.07
- ✘ `fire_catching__564219s18` CLAP 0.434, PQ 7.77, RMS 0.011 — AST: слышно Cap gun 0.36, а Crackle только 0.01. AST: Cap gun 0.36, Gunshot, gunfire 0.11, Arrow 0.10
- ✘ `fire_catching__564219s17` CLAP 0.409, PQ 7.17, RMS 0.016 — AST: слышно Cap gun 0.15, а Crackle только 0.01. AST: Cap gun 0.15, Chop 0.06, Scissors 0.03
- ✔ `fire_catching__564219s11` CLAP 0.404, PQ 7.27, RMS 0.010 — AST: слышно Cap gun 0.35, а Crackle только 0.01. AST: Cap gun 0.35, Camera 0.13, Gunshot, gunfire 0.12
- ✘ `fire_catching__564219s15` CLAP 0.390, PQ 7.09, RMS 0.018 — AST: слышно Shuffling cards 0.28, а Crackle только 0.00. AST: Shuffling cards 0.28, Chop 0.15, Cap gun 0.02
- ✘ `fire_catching__564219s13` CLAP 0.359, PQ 7.38, RMS 0.014 — AST: слышно Cap gun 0.25, а Crackle только 0.00. AST: Cap gun 0.25, Whip 0.05, Chop 0.05
- ✘ `fire_catching__564219s14` CLAP 0.351, PQ 7.40, RMS 0.015 — AST: слышно Shuffling cards 0.19, а Crackle только 0.00. AST: Shuffling cards 0.19, Scissors 0.04, Chop 0.04
- ✘ `fire_catching__564219s12` CLAP 0.327, PQ 7.69, RMS 0.014 — AST: слышно Cap gun 0.83, а Crackle только 0.00. AST: Cap gun 0.83, Gunshot, gunfire 0.41, Single-lens reflex camera 0.02
