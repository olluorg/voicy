# Машинный отбор

| элемент | выбран | CLAP | PQ | забраковано |
|---|---|---|---|---|
| winter_outside | dsp21 | 0.385 | 5.89 | 4 из 6 |
| hut_air | 39b9acs12 | 0.132 | 5.62 | 2 из 4 |
| fire_burning | d0ee6as14 | 0.386 | 6.60 | 0 из 4 |
| door_open | e030b5s15 | 0.224 | 6.34 | 4 из 8 |
| door_close | be1c2bs16 | 0.279 | 7.70 | 0 из 8 |
| floor_walk | d2f72as14 | 0.269 | 5.85 | 3 из 4 |
| table_thud | 9565fas14 | 0.198 | 6.65 | 1 из 4 |
| cough | fc91c4s14 | 0.313 | 6.06 | 0 из 4 |
| fire_catching | d04152s11 | 0.280 | 6.32 | 1 из 4 |

## winter_outside — цель: fierce winter wind and blizzard outside | strong winter blizzard wind blowing
- ✔ `winter_outside__dsp21` CLAP 0.385, PQ 5.89, RMS 0.065 — годен. AST: Eruption 0.12, White noise 0.07, Waves, surf 0.06
- · `winter_outside__dsp22` CLAP 0.373, PQ 5.79, RMS 0.050 — годен. AST: Eruption 0.13, White noise 0.07, Field recording 0.07
- ✘ `winter_outside__f14446s11` CLAP 0.363, PQ 4.56, RMS 0.056 — AST: mic_wind 0.38. AST: Wind noise (microphone) 0.38, Wind 0.31, Boat, Water vehicle 0.19
- ✘ `winter_outside__f14446s14` CLAP 0.351, PQ 4.82, RMS 0.051 — AST: mic_wind 0.31. AST: Wind noise (microphone) 0.31, Wind 0.24, Eruption 0.13
- ✘ `winter_outside__f14446s12` CLAP 0.346, PQ 4.76, RMS 0.037 — AST: mic_wind 0.58. AST: Wind noise (microphone) 0.58, Wind 0.53, Ocean 0.35
- ✘ `winter_outside__f14446s13` CLAP 0.302, PQ 4.56, RMS 0.049 — AST: mic_wind 0.46. AST: Wind noise (microphone) 0.46, Wind 0.28, Rustling leaves 0.16

## hut_air — цель: still cold air in a wooden room | the quiet interior of a room
- ✔ `hut_air__39b9acs12` CLAP 0.132, PQ 5.62, RMS 0.011 — годен. AST: Stomach rumble 0.39, Snake 0.09, Animal 0.04
- ✘ `hut_air__39b9acs13` CLAP 0.101, PQ 6.13, RMS 0.011 — CLAP: «a cat meowing» 0.79. AST: Patter 0.06, Inside, small room 0.03, Snake 0.03
- · `hut_air__39b9acs11` CLAP 0.091, PQ 5.92, RMS 0.009 — годен. AST: Patter 0.22, Rodents, rats, mice 0.06, Stomach rumble 0.05
- ✘ `hut_air__39b9acs14` CLAP 0.040, PQ 5.65, RMS 0.009 — CLAP: «a cat meowing» 1.00. AST: Walk, footsteps 0.07, Silence 0.03, Patter 0.03

## fire_burning — цель: wood fire crackling in a stove | a wood fire crackling
- ✔ `fire_burning__d0ee6as14` CLAP 0.386, PQ 6.60, RMS 0.013 — годен. AST: Finger snapping 0.16, Scissors 0.11, Hands 0.06
- · `fire_burning__d0ee6as11` CLAP 0.310, PQ 5.98, RMS 0.019 — годен. AST: Finger snapping 0.12, Hands 0.06, Clapping 0.05
- · `fire_burning__d0ee6as13` CLAP 0.305, PQ 5.39, RMS 0.019 — годен. AST: Finger snapping 0.21, Speech 0.08, Insect 0.03
- · `fire_burning__d0ee6as12` CLAP 0.324, PQ 4.03, RMS 0.016 — годен. AST: Finger snapping 0.41, Speech 0.26, Fire 0.07

## door_open — цель: heavy wooden door creaking open; ждём: Door, Creak, Squeak
- ✔ `door_open__e030b5s15` CLAP 0.224, PQ 6.34, RMS 0.107 — годен. AST: Door 0.27, Knock 0.24, Sliding door 0.10
- · `door_open__e030b5s13` CLAP 0.216, PQ 6.44, RMS 0.080 — годен. AST: Door 0.55, Sliding door 0.20, Knock 0.20
- · `door_open__e030b5s11` CLAP 0.190, PQ 6.67, RMS 0.070 — годен. AST: Knock 0.16, Door 0.14, Music 0.10
- ✘ `door_open__e030b5s14` CLAP 0.169, PQ 5.13, RMS 0.091 — AST: engine 0.31; AST: слышно Vehicle 0.31, а Door только 0.06. AST: Vehicle 0.31, Rumble 0.22, Car 0.09
- ✘ `door_open__e030b5s16` CLAP 0.146, PQ 5.94, RMS 0.080 — AST: слышно Scrape 0.10, а Door только 0.01. AST: Sound effect 0.18, Scrape 0.10, Rattle 0.10
- · `door_open__e030b5s18` CLAP 0.152, PQ 5.51, RMS 0.097 — годен. AST: Door 0.18, Typewriter 0.15, Typing 0.13
- ✘ `door_open__e030b5s12` CLAP 0.104, PQ 7.37, RMS 0.073 — AST: слышно Knock 0.30, а Door только 0.17. AST: Knock 0.30, Typing 0.23, Computer keyboard 0.20
- ✘ `door_open__e030b5s17` CLAP 0.127, PQ 5.26, RMS 0.113 — AST: engine 0.23; CLAP: «hail hitting a roof» 0.50. AST: Door 0.24, Vehicle 0.23, Typewriter 0.18

## door_close — цель: heavy wooden door creaking shut; ждём: Door, Creak, Thump, thud
- ✔ `door_close__be1c2bs16` CLAP 0.279, PQ 7.70, RMS 0.074 — годен. AST: Door 0.42, Knock 0.23, Speech 0.15
- · `door_close__be1c2bs11` CLAP 0.274, PQ 7.48, RMS 0.072 — годен. AST: Door 0.53, Speech 0.12, Slam 0.12
- · `door_close__be1c2bs15` CLAP 0.258, PQ 7.32, RMS 0.073 — годен. AST: Door 0.47, Domestic animals, pets 0.18, Animal 0.18
- · `door_close__be1c2bs17` CLAP 0.262, PQ 7.11, RMS 0.089 — годен. AST: Door 0.40, Thunk 0.25, Knock 0.06
- · `door_close__be1c2bs18` CLAP 0.252, PQ 7.44, RMS 0.101 — годен. AST: Door 0.56, Thunk 0.29, Slam 0.20
- · `door_close__be1c2bs14` CLAP 0.241, PQ 7.69, RMS 0.092 — годен. AST: Door 0.63, Slam 0.23, Thunk 0.16
- · `door_close__be1c2bs13` CLAP 0.226, PQ 7.63, RMS 0.097 — годен. AST: Door 0.62, Slam 0.22, Thunk 0.14
- · `door_close__be1c2bs12` CLAP 0.185, PQ 7.55, RMS 0.058 — годен. AST: Single-lens reflex camera 0.19, Door 0.15, Gunshot, gunfire 0.10

## floor_walk — цель: footsteps on wooden floorboards; ждём: Walk, footsteps, Creak
- ✘ `floor_walk__d2f72as12` CLAP 0.311, PQ 6.34, RMS 0.014 — AST: слышно Heart murmur 0.04, а Walk, footsteps только 0.02. AST: Sound effect 0.05, Heart murmur 0.04, Scrape 0.04
- ✘ `floor_walk__d2f72as13` CLAP 0.285, PQ 5.72, RMS 0.023 — AST: слышно Speech 0.21, а Walk, footsteps только 0.10. AST: Speech 0.21, Walk, footsteps 0.10, Scrape 0.07
- ✔ `floor_walk__d2f72as14` CLAP 0.269, PQ 5.85, RMS 0.039 — годен. AST: Walk, footsteps 0.14, Stomach rumble 0.08, Crackle 0.08
- ✘ `floor_walk__d2f72as11` CLAP 0.274, PQ 5.33, RMS 0.027 — AST: слышно Stomach rumble 0.38, а Walk, footsteps только 0.03. AST: Stomach rumble 0.38, Scrape 0.05, Walk, footsteps 0.03

## table_thud — цель: heavy object thudding on a table; ждём: Thump, thud, Thunk, Knock
- ✔ `table_thud__9565fas14` CLAP 0.198, PQ 6.65, RMS 0.052 — годен. AST: Knock 0.46, Thunk 0.21, Door 0.11
- · `table_thud__9565fas13` CLAP 0.128, PQ 6.98, RMS 0.047 — годен. AST: Knock 0.12, Sound effect 0.08, Thunk 0.02
- · `table_thud__9565fas11` CLAP 0.112, PQ 6.47, RMS 0.063 — годен. AST: Knock 0.74, Sound effect 0.07, Door 0.05
- ✘ `table_thud__9565fas12` CLAP 0.100, PQ 6.40, RMS 0.054 — AST: слышно Coin (dropping) 0.04, а Thunk только 0.03. AST: Sound effect 0.19, Coin (dropping) 0.04, Bang 0.03

## cough — цель: deep rasping cough; ждём: Cough, Throat clearing
- ✔ `cough__fc91c4s14` CLAP 0.313, PQ 6.06, RMS 0.043 — годен. AST: Throat clearing 0.50, Cough 0.07, Speech 0.02
- · `cough__fc91c4s12` CLAP 0.278, PQ 6.43, RMS 0.031 — годен. AST: Throat clearing 0.77, Cough 0.08, Speech 0.06
- · `cough__fc91c4s13` CLAP 0.222, PQ 5.54, RMS 0.040 — годен. AST: Throat clearing 0.55, Cough 0.06, Speech 0.05
- · `cough__fc91c4s11` CLAP 0.198, PQ 5.80, RMS 0.042 — годен. AST: Throat clearing 0.75, Cough 0.08, Speech 0.04

## fire_catching — цель: fire starting and growing; ждём: Fire, Crackle
- ✔ `fire_catching__d04152s11` CLAP 0.280, PQ 6.32, RMS 0.007 — годен. AST: Fire 0.21, Arrow 0.06, Silence 0.04
- ✘ `fire_catching__d04152s14` CLAP 0.255, PQ 6.19, RMS 0.004 — AST: слышно Arrow 0.20, а Fire только 0.06. AST: Arrow 0.20, Rain 0.07, Fire 0.06
- · `fire_catching__d04152s12` CLAP 0.253, PQ 6.25, RMS 0.009 — годен. AST: Fire 0.23, Firecracker 0.20, Fireworks 0.07
- · `fire_catching__d04152s13` CLAP 0.153, PQ 6.08, RMS 0.008 — годен. AST: Fire 0.16, Rain 0.12, Rain on surface 0.06
