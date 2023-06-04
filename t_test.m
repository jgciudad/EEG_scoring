

ST_blab_S1_noArts=readtable("C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SleepTransformer_mice\evaluation\t_test\BLAB_S1_t_test_data_ST_noArts.xlsx");
ST_blab_S1_noArts=ST_blab_S1_noArts.acc;

ST_blab_S2_noArts=readtable("C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SleepTransformer_mice\evaluation\t_test\BLAB_S2_t_test_data_ST_noArts.xlsx");
ST_blab_S2_noArts=ST_blab_S2_noArts.acc;

ST_blab_S1_Arts=readtable("C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SleepTransformer_mice\evaluation\t_test\BLAB_S1_t_test_data_ST_withArts.xlsx");
ST_blab_S1_Arts=ST_blab_S1_Arts.acc;

ST_blab_S2_Arts=readtable("C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SleepTransformer_mice\evaluation\t_test\BLAB_S2_t_test_data_ST_withArts.xlsx");
ST_blab_S2_Arts=ST_blab_S2_Arts.acc;

ST_blab_noArts= (ST_blab_S1_noArts+ST_blab_S2_noArts)/2;
ST_blab_Arts = (ST_blab_S1_Arts+ST_blab_S2_Arts)/2;


ST_kornum_noArts = readtable("C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SleepTransformer_mice\evaluation\t_test\KORNUMt_test_data_ST_noArts.xlsx");
ST_kornum_noArts = ST_kornum_noArts.acc;

ST_kornum_Arts = readtable("C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SleepTransformer_mice\evaluation\t_test\KORNUMt_test_data_ST_withArts.xlsx");
ST_kornum_Arts = ST_kornum_Arts.acc;


SP_kornum_noArts = readtable("C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\kornum_data\t_test\Ttest_SPINDLE_CNN1_KORNUM.csv");
SP_kornum_noArts = SP_kornum_noArts.acc;

SP_kornum_Arts = readtable("C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\kornum_data\t_test\Ttest_SPINDLE_whole_KORNUM.csv");
SP_kornum_Arts=SP_kornum_Arts.acc;


SP_blab_S1_noArts = readtable("C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\spindle_data\t_test\Ttest_SPINDLE_CNN1_BrownLab_s1.csv");
SP_blab_S1_noArts=SP_blab_S1_noArts.acc;

SP_blab_S2_noArts = readtable("C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\spindle_data\t_test\Ttest_SPINDLE_CNN1_BrownLab_s2.csv");
SP_blab_S2_noArts=SP_blab_S2_noArts.acc;

SP_blab_noArts = (SP_blab_S1_noArts+SP_blab_S2_noArts)/2;


SP_blab_S2_Arts = readtable("C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\spindle_data\t_test\Ttest_SPINDLE_whole_BrownLab_s2.csv");
SP_blab_S2_Arts = SP_blab_S2_Arts.acc;

SP_blab_S1_Arts = readtable("C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SPINDLE\spindle_data\t_test\Ttest_SPINDLE_whole_BrownLab_s1.csv");
SP_blab_S1_Arts = SP_blab_S1_Arts.acc;

SP_blab_Arts=(SP_blab_S1_Arts+SP_blab_S2_Arts)/2;



[h1, p1, ci1] = ttest(SP_kornum_noArts,ST_kornum_noArts);
[h2, p2, ci2] = ttest(SP_blab_noArts,ST_blab_noArts);
[h3, p3, ci3] = ttest(SP_kornum_Arts,ST_kornum_Arts);
[h4, p4, ci4] = ttest(SP_blab_Arts,ST_blab_Arts);

sd_a = std(SP_kornum_noArts);
sd_b = std(ST_kornum_noArts);
sd_c = std(SP_blab_noArts);
sd_d = std(ST_blab_noArts);
sd_e = std(SP_kornum_Arts);
sd_f = std(ST_kornum_Arts);
sd_g = std(SP_blab_Arts);
sd_h = std(ST_blab_Arts);
