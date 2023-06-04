function collection = aggregate_sleeptransformer(nchan)
    
    nchan=1
    Nfold = 1;
    yh = cell(Nfold,1);
    yt = cell(Nfold,1);
    mat_path = 'C:\Users\javig\Documents\THESIS_DATA\SleepTransformer_mice\data_preprocessing\spindle_data\mat\scorer_2\';
    listing = dir([mat_path, '*_eeg1.mat']);
    load("C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SleepTransformer_mice\data_preprocessing\spindle_data\data_split_eval.mat");
    
    acc_novote = [];
    
    seq_len = 21;
    for fold = 1 : Nfold
        fold
        %test_s = test_sub{fold};
        test_s = test_sub;
        sample_size = zeros(numel(test_s), 1);
        for i = 1 : numel(test_s)
            i
            sname = listing(test_s(i)).name;
            load([mat_path,sname], 'label');
            % handle the different here
            sample_size(i) = numel(label) -  (seq_len - 1); 
            yt{fold} = [yt{fold}; double(label)];
        end
        
        if(seq_len < 100)
            load("C:\Users\javig\Documents\Drive\DTU\MASTER THESIS\Code\EEG_scoring\SleepTransformer_mice\scratch_training\sleeptransformer\scratch_training_3chan_CPU_test2\n1\SPINDLE_evaluation\scorer_2\test_ret.mat");
        else
            load(['./intepretable_sleep/sleeptransformer_simple_longseq/scratch_training_',num2str(nchan),'chan/n',num2str(fold),'/test_ret.mat']);
        end
        
        
        acc_novote = [acc_novote; acc];
        
        score_ = cell(1,seq_len);
        for n = 1 : seq_len
            score_{n} = softmax(squeeze(score(:,n,:)));
        end
        score = score_;
        clear score_;
        
        sz = [4 14];
        varTypes = ["char","double","double","double","double","double","double","double","double","double","double","double","double","double"];
        varNames = ["filename","acc","prec_N","sens_N","f1_N","prec_R","sens_R","f1_R","prec_W","sens_W","f1_W","prec_A","sens_A","f1_A"];
        t_test_data = table('Size',sz,'VariableTypes',varTypes,'VariableNames',varNames);
        for i = 1 : numel(test_s)
            sname = listing(test_s(i)).name;
            label = load([mat_path,sname], 'label');
            yt_i = label.label;

            start_pos = sum(sample_size(1:i-1)) + 1;
            end_pos = sum(sample_size(1:i-1)) + sample_size(i);
            score_i = cell(1,seq_len);
            %valid_ind = cell(1,seq_len);
            for n = 1 : seq_len
                score_i{n} = score{n}(start_pos:end_pos, :);
                N = size(score_i{n},1);
                %valid_ind{n} = ones(N,1);

                score_i{n} = [ones(seq_len-1,4); score{n}(start_pos:end_pos, :)];
                %valid_ind{n} = [zeros(seq_len-1,1); valid_ind{n}]; 
                score_i{n} = circshift(score_i{n}, -(seq_len - n), 1);
                %valid_ind{n} = circshift(valid_ind{n}, -(seq_len - n), 1);
            end

            smoothing = 0;
            %fused_score = score_i{1};
            %fused_score = log(score_i{1}.*repmat(valid_ind{1},1,5));
            fused_score = log(score_i{1});
            for n = 2 : seq_len
                if(smoothing == 0)
                    %fused_score = fused_score + log(score_i{n}.*repmat(valid_ind{n},1,5));
                    fused_score = fused_score + log(score_i{n});
                else
                    %fused_score = fused_score + score_i{n}.*repmat(valid_ind{n},1,5);
                    fused_score = fused_score + score_i{n};
                end
            end

            yhat = zeros(1,size(fused_score,1));
            for k = 1 : size(fused_score,1)
                [~, yhat(k)] = max(fused_score(k,:));
            end
            
            
            yt_i = double(yt_i);
            yh_i = double(yhat');
            
%             yh_i = yh_i(yt_i~=4); % filter out artifacts
%             yt_i = yt_i(yt_i~=4); % filter out artifacts
            
            acc = sum(yh_i == yt_i)/numel(yt_i);

            [mysensitivity_i, myselectivity_i]  = calculate_sensitivity_selectivity(yt_i, yh_i); % THEY MADE A MISTAKE, THIS IS NOT SELECTIVITY, IS PRECISION!!! (is actually good because is what I want)

            [fscore_i, ~, ~] = litis_class_wise_f1(yt_i, yh_i);

            t_test_data(i,:) = {sname,acc,myselectivity_i(2),mysensitivity_i(2),fscore_i(2),myselectivity_i(3),mysensitivity_i(3),fscore_i(3),myselectivity_i(1),mysensitivity_i(1),fscore_i(1),myselectivity_i(4),mysensitivity_i(4),fscore_i(4)};
            
            a=9;
            
        end
        
    writetable(t_test_data,"BLAB_S2_t_test_data_ST_withArts.xlsx")
    writetable(t_test_data,"BLAB_S2_t_test_data_ST_withArts")

    end

    