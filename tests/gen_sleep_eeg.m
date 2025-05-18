% gen_sleep_eeg.m


fs = 100;                         % Sampling rate
duration_min = 120;                % Total duration (minutes)
total_samples = duration_min * 60 * fs;
epoch_sec = 30;
samples_per_epoch = epoch_sec * fs;
n_epochs = total_samples / samples_per_epoch;

% === Initialize EEG ===
EEG = eeg_emptyset;
EEG.setname = 'synthetic_sleep_eeg';
EEG.nbchan = 3;
EEG.srate = fs;
EEG.trials = 1;
EEG.pnts = total_samples;
EEG.xmin = 0;
EEG.xmax = (EEG.pnts - 1) / EEG.srate;
EEG.times = 1 * EEG.pnts;
EEG.data = randn(EEG.nbchan, EEG.pnts);  % Base EEG signal

% === Channel Labels ===
labels = {'C3', 'C4', 'O1'};
for i = 1:EEG.nbchan
    EEG.chanlocs(i).labels = labels{i};
end

single_stages = {'W', '1', 'R'};
stage_pool = [repmat(single_stages, 1, 4), repmat({'2-3'}, 1, 4)];
shuffled = stage_pool(randperm(length(stage_pool)));

stages = {};
for i = 1:length(shuffled)
    if strcmp(shuffled{i}, '2-3')
        stages = [stages; '2'; '3'];
    else
        stages = [stages; shuffled{i}];
    end
end
EEG.etc.stages = stages;

% === Stage Label Mapping ===
stage_map = containers.Map({'W', '1', '2', '3', 'R'}, {'Wake', 'N1', 'N2', 'N3', 'REM'});

% === Insert Stage Events ===
EEG.event = struct('latency', {}, 'duration', {}, 'type', {}, 'id', {}, 'is_reject', {});
for i = 1:length(stages)
    EEG.event(i).latency = (i-1)*samples_per_epoch + 1;
    EEG.event(i).duration = samples_per_epoch;
    EEG.event(i).type = stage_map(stages{i});
    EEG.event(i).id = i;
    EEG.event(i).is_reject = 0;
end

% === Insert Arousal Events ===
n_arousals = 5;
min_arousal_samples = 3 * fs;  % at least 3 seconds
max_arousal_samples = 6 * fs;

used_indices = [];
for i = 1:n_arousals
    while true
        latency = randi([1, EEG.pnts - max_arousal_samples]);
        if all(abs(latency - used_indices) > max_arousal_samples)
            break;
        end
    end
    used_indices(end+1) = latency;

    duration = randi([min_arousal_samples, max_arousal_samples]);

    % Add artifact signal to EEG
    EEG.data(:, latency:latency+duration-1) = EEG.data(:, latency:latency+duration-1) + ...
                                               5 * randn(EEG.nbchan, duration);  % add strong noise

    % Define arousal event
    arousal_event.latency = latency;
    arousal_event.duration = duration;
    arousal_event.type = 'Arousal 3 ARO SPONT';
    arousal_event.id = length(EEG.event) + 1;
    arousal_event.is_reject = 1;   % <==== Marked as reject
    EEG.event(end + 1) = arousal_event;
end

% === Metadata ===
EEG.etc.eeglabvers = '2022.0';
EEG.etc.amp_startdate = '2024-10-28T20:00:00';
EEG.etc.rec_startdate = '2024-10-28T22:08:00';

% === Save File ===
EEG = eeg_checkset(EEG);
%pop_saveset(EEG, 'filename', 'synthetic_sleep_eeg.set', 'filepath', pwd);

