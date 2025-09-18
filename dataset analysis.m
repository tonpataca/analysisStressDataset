%% ============================================================
%  Feature extraction (ECG, EDA, Temp °C) with windowing + Per-Video HRV
%  Robust temperature:
%   - Auto-pick column (Temp/TMP/°C preferred; else scans AI1..AI5)
%   - Auto-detect units: C | counts | V | mV
%   - BITalino TMP transfer function for counts/V/mV
%   - Optional re-centering so medians ≈ 35 °C
%
%  Outputs:
%     - features_master_all.csv
%     - emotion_<Emotion>.csv
%     - features_<Subject>.xlsx   (All + per-emotion sheets + VideoHRV)
%     - video_hrv_<Subject>.csv
%     - video_hrv_master_all.csv
%     - video_hrv_emotion_<Emotion>.csv
%  ============================================================

clear; clc;

%% ===================== CONFIG ===============================
dataDir    = "C:\Users\msi-pc\Desktop\Stress\dataset";   % <-- CHANGE
outDir     = fullfile(dataDir, "features_out7");
fs_default = 100;                                        % fallback Hz

% Windowing (per-window features)
win_sec      = 10;
overlap_sec  = 5;      % hop = win_sec - overlap_sec
save_per_subject_excel = true;

% BITalino TMP conversion defaults
VCC_TMP   = 3.3;   % operating voltage
NBITS_TMP = 10;    % ADC resolution

% Optional: force human skin range by re-centering median to 35 °C
FORCE_SKIN_RANGE = true;    % <--- set false if you don't want re-centering
TARGET_MED_C     = 35.0;    % target median
PLAUS_RANGE      = [10 45]; % hard sanity

% Emotion labels for videos 1..18
emotionNames = [ ...
    "Neutral","Neutral","Positive/Relaxing","Sadness","Fear/Anxiety","Fear/Anxiety", ...
    "Happiness","Social Stress","Angry/Irritation","Humor","Shame/Guilt","Intense Fear", ...
    "Satisfaction/Love","Interpersonal Conflict","Surprise","Sadness","Neutral","Neutral" ...
];

% Ensure output directory exists
if exist(outDir, 'dir') ~= 7
    [ok,msg] = mkdir(outDir);
    assert(ok, "Could not create outDir: %s", msg);
end

%% ===================== PROCESS ALL SUBJECTS =================
subjFolders = dir(dataDir);
subjFolders = subjFolders([subjFolders.isdir]);
subjFolders = subjFolders(~ismember({subjFolders.name},{'.','..'}));

allRows          = table();  % per-window master
allVideoHRVRows  = table();  % per-video master

for s = 1:numel(subjFolders)
    subjPath = fullfile(subjFolders(s).folder, subjFolders(s).name);
    subjName = subjFolders(s).name;
    fprintf('\n=== Subject: %s ===\n', subjName);

    rows_win  = table();
    rows_vhrv = table();

    for v = 1:18
        fName = fullfile(subjPath, sprintf("bitalino_%d.txt", v));
        if ~isfile(fName)
            warning('Missing file: %s', fName);
            continue;
        end

        [T, fs] = readBitalinoTxt(fName, fs_default);

        % Make sure channels are numeric
        for nm = intersect(T.Properties.VariableNames, ["AI1","AI2","AI3","AI4","AI5"])
            T.(nm) = toNumeric(T.(nm));
        end

        % ---------- Temperature: auto-find column + auto-convert to °C ----------
        [T.Temp_C, temp_col, temp_mode, temp_med, recentered_by] = ...
            pick_and_convert_temperature(T, VCC_TMP, NBITS_TMP, FORCE_SKIN_RANGE, TARGET_MED_C, PLAUS_RANGE);

        % Quick QC printout
        mnT = min(T.Temp_C,[],'omitnan'); mxT = max(T.Temp_C,[],'omitnan');
        recStr = "";
        if ~isnan(recentered_by) && recentered_by~=0
            recStr = sprintf(" | recentered %+0.2f °C", recentered_by);
        end
        fprintf('TEMP %s | col=%s mode=%s | min/med/max=%.2f/%.2f/%.2f °C%s\n', ...
            fName, string(temp_col), string(temp_mode), mnT, temp_med, mxT, recStr);

        if ~(isfinite(temp_med) && temp_med>=PLAUS_RANGE(1) && temp_med<=PLAUS_RANGE(2))
            warning('Implausible median temperature (%.2f °C). Verify temp channel/units for this file.', temp_med);
        end

        % ---------- Labels ----------
        emoLabel   = emotionNames(v);
        classLabel = emotionalClass(emoLabel);  % "Emotional" or "Neutral"

        %% ---------- (A) Per-window features ----------
        hop   = max(1, round((win_sec - overlap_sec) * fs));
        winLen = round(win_sec * fs);

        if height(T) < max(1, fs*win_sec)
            R = nanFeatureRow();
            R.subject     = subjName;  R.video_index = v;  R.emotion = emoLabel;
            R.class_label = classLabel; R.win_start_s = 0; R.win_end_s = height(T)/fs;
            rows_win = [rows_win; struct2table(R)]; %#ok<AGROW>
        else
            lastStart = height(T) - winLen + 1;
            starts    = 1:hop:lastStart;

            for st = starts
                seg = T(st:st+winLen-1, :);
                R   = featuresOneWindow(seg, fs);

                R.subject     = subjName;
                R.video_index = v;
                R.emotion     = emoLabel;
                R.class_label = classLabel;
                R.win_start_s = (st-1)/fs;
                R.win_end_s   = (st+winLen-2)/fs;
                rows_win = [rows_win; struct2table(R)]; %#ok<AGROW>
            end
        end

        %% ---------- (B) Per-video HRV (whole video) ----------
        ecg = []; if ismember('AI2', T.Properties.VariableNames), ecg = T.AI2(:); end
        if isempty(ecg) || all(~isfinite(ecg)) || height(T) < fs*20
            RV = nanVideoHRVRow();
        else
            RV = hrvFromWholeSignal(ecg, fs);
        end

        RV.subject     = subjName;
        RV.video_index = v;
        RV.emotion     = emoLabel;
        RV.class_label = classLabel;
        RV.duration_s  = height(T)/fs;

        % Per-video temperature summary (with smoothing + robust slope)
        if any(isfinite(T.Temp_C))
            tmin = (0:height(T)-1)'/fs/60;
            tmp  = fillTemp(T.Temp_C);
            tmp  = smoothTemp(tmp, fs, 2.0);
            [slope, meanC, medC, iqrC] = robustTempStats(tmin, tmp);
            RV.Temp_mean_C          = meanC;
            RV.Temp_slope_C_per_min = slope;
            RV.Temp_median_C        = medC;
            RV.Temp_iqr_C           = iqrC;
            RV.Temp_mode            = string(temp_mode);
            RV.Temp_vref_V          = VCC_TMP;
        end

        rows_vhrv = [rows_vhrv; struct2table(RV)]; %#ok<AGROW>
    end

    %% --------- Save per-subject (Excel + per-video CSV) ---------
    if ~isempty(rows_win) && save_per_subject_excel
        outXlsx = fullfile(outDir, sprintf("features_%s.xlsx", subjName));
        if exist(outXlsx,'file'), delete(outXlsx); end

        writetable(rows_win,  outXlsx, 'Sheet', 'All');      % per-window
        emos_subj = unique(rows_win.emotion,'stable');
        for j = 1:numel(emos_subj)
            emo = emos_subj(j);
            tbl = rows_win(rows_win.emotion==emo, :);
            writetable(tbl, outXlsx, 'Sheet', sheetname(emo));
        end
        writetable(rows_vhrv, outXlsx, 'Sheet', 'VideoHRV'); % per-video
        fprintf('Saved per-subject Excel: %s\n', outXlsx);
    end

    if ~isempty(rows_vhrv)
        outCSVv = fullfile(outDir, sprintf("video_hrv_%s.csv", subjName));
        writetable(rows_vhrv, outCSVv);
        fprintf('Saved per-subject VIDEO HRV: %s\n', outCSVv);
    end

    allRows         = [allRows; rows_win];          %#ok<AGROW>
    allVideoHRVRows = [allVideoHRVRows; rows_vhrv]; %#ok<AGROW>
end

%% --------- Master per-window CSV + emotion splits ----------
masterCsv = fullfile(outDir, "features_master_all.csv");
writetable(allRows, masterCsv);
fprintf('Saved master per-window: %s\n', masterCsv);

if ~isempty(allRows)
    emos = unique(allRows.emotion, 'stable');
    for i = 1:numel(emos)
        emo = emos(i);
        mask = allRows.emotion == emo;
        tbl  = allRows(mask, :);
        fname = fullfile(outDir, sprintf("emotion_%s.csv", slugify(emo)));
        writetable(tbl, fname);
        fprintf('Saved per-window emotion file: %s  (rows=%d)\n', fname, height(tbl));
    end
end

%% --------- Master per-video HRV CSV + emotion splits ----------
masterVidCsv = fullfile(outDir, "video_hrv_master_all.csv");
writetable(allVideoHRVRows, masterVidCsv);
fprintf('Saved master per-video HRV: %s\n', masterVidCsv);

if ~isempty(allVideoHRVRows)
    emos = unique(allVideoHRVRows.emotion, 'stable');
    for i = 1:numel(emos)
        emo = emos(i);
        mask = allVideoHRVRows.emotion == emo;
        tbl  = allVideoHRVRows(mask, :);
        fname = fullfile(outDir, sprintf("video_hrv_emotion_%s.csv", slugify(emo)));
        writetable(tbl, fname);
        fprintf('Saved per-video HRV emotion file: %s  (rows=%d)\n', fname, height(tbl));
    end
end

fprintf('\nDone. Outputs in %s\n', outDir);

%% ===================== HELPERS ==============================
function [T, fs] = readBitalinoTxt(fpath, fs_default)
fs = fs_default;

% sniff header for sampling rate + first numeric row
lines = readlines(fpath);
dataStart = 1;
maxScan = min(200, numel(lines));
for i = 1:maxScan
    L = strtrim(lines(i));
    if contains(L,"Sampling Rate") || contains(L,"SamplingRate") || contains(L,"Sampling Rate (Hz)")
        tok = regexp(L,'(\d+)\s*Hz','tokens','once');
        if ~isempty(tok), fs = str2double(tok{1}); end
    end
    if ~isempty(regexp(L,'^\s*\d+(\s*,\s*[\d\.\-Ee]+)+\s*$', 'once'))
        dataStart = i; break;
    end
end

opts = detectImportOptions(fpath, ...
    'NumHeaderLines', max(0, dataStart-1), ...
    'Delimiter', ',', ...
    'ConsecutiveDelimitersRule', 'join', ...
    'ExtraColumnsRule', 'ignore', ...
    'Whitespace', ' \b\t');
try, opts = setvaropts(opts, opts.VariableNames, 'Type', 'string'); catch, end
T = readtable(fpath, opts);

if ~strcmp(T.Properties.VariableNames{1}, 'Timestamp')
    T.Properties.VariableNames{1} = 'Timestamp';
end

% rename last up-to-5 columns to AI1..AI5
nC = width(T);
last5 = max(1, nC-4):nC;
aNamesCell = {'AI1','AI2','AI3','AI4','AI5'};
numToRename = min(numel(last5), numel(aNamesCell));
for k = 1:numToRename
    T.Properties.VariableNames{ last5(k) } = aNamesCell{k};
end

% keep as many AIs as available
want = {'Timestamp','AI1','AI2','AI3','AI4','AI5'};
keep = intersect(want, T.Properties.VariableNames, 'stable');
T = T(:, keep);

% force numeric
for c = 1:width(T)
    nm = T.Properties.VariableNames{c};
    col = T{:, c};
    if isnumeric(col), T{:, c} = double(col); continue; end
    s = toStringVector(col); vals = str2double(s);
    if all(isnan(vals) & s ~= "")
        if strcmp(nm, 'Timestamp')
            dt = tryParseDatetime(s);
            if any(~isnat(dt))
                t0idx = find(~isnat(dt),1,'first');
                tsec = seconds(dt - dt(t0idx));
                vals = double(tsec);
            end
        end
    end
    if all(isnan(vals))
        if strcmp(nm, 'Timestamp'), T{:, c} = (0:height(T)-1)'; 
        else, T{:, c} = nan(height(T),1);
        end
    else
        T{:, c} = vals;
    end
end
end

function [Temp_C, col_used, mode_used, med_used, recentered_by] = ...
    pick_and_convert_temperature(T, Vcc, nbits, force_skin, target_med, plaus)
% Scan likely temperature columns (Temp/TMP/°C, then AI1..AI5), try C|counts|V|mV,
% pick the interpretation closest to target_med (penalize outside plaus range).
% Optionally re-center by a constant offset to target_med.

cols = T.Properties.VariableNames;
% rank by name first
scoreName = zeros(1,numel(cols));
for i=1:numel(cols)
    nm = lower(cols{i});
    if any(contains(nm, ["temp","tmp","celsius","°c","ºc"])), scoreName(i)=scoreName(i)+3; end
    if any(strcmp(nm, ["ai4","ai3","ai2","ai1","ai5"])),    scoreName(i)=scoreName(i)+1; end
end
[~,ord] = sort(scoreName,'descend');
cols = cols(ord);

best = struct('sc',inf,'col',"NA",'mode',"NA",'Ti',[],'med',NaN);

for i=1:numel(cols)
    nm = cols{i}; if strcmpi(nm,'Timestamp'), continue; end
    x = double(T.(nm)(:));
    if ~any(isfinite(x)), continue; end

    cands = {};
    % As Celsius (already)
    cands{end+1} = struct('mode',"C",   'Ti',x); %#ok<AGROW>

    % As counts
    if max(x) <= (2^nbits - 1) + 1e-6 && min(x) >= -1
        V = (x./(2^nbits))*Vcc;
        cands{end+1} = struct('mode',"counts",'Ti',(V - 0.5)*100);
    end
    % As Volts
    if max(x) <= Vcc + 0.3 && min(x) >= -0.3
        cands{end+1} = struct('mode',"V", 'Ti',(x - 0.5)*100);
    end
    % As mV
    if max(x) <= 1000*Vcc + 100 && min(x) >= -100
        cands{end+1} = struct('mode',"mV",'Ti',(x/1000 - 0.5)*100);
    end

    for k=1:numel(cands)
        Ti   = cands{k}.Ti;
        medT = median(Ti,'omitnan');
        inR  = isfinite(medT) && medT>=plaus(1) && medT<=plaus(2);
        sc   = abs(medT - target_med) + (~inR)*100; % heavy penalty if outside 10–45
        if sc < best.sc
            best.sc  = sc;
            best.col = nm;
            best.mode= cands{k}.mode;
            best.Ti  = Ti;
            best.med = medT;
        end
    end
end

if isempty(best.Ti)
    Temp_C = nan(height(T),1); col_used="NA"; mode_used="NA"; med_used=NaN; recentered_by=NaN; return;
end

Temp_C      = best.Ti;
col_used    = string(best.col);
mode_used   = string(best.mode);
med_used    = best.med;
recentered_by = 0;

% Optional: re-center by constant shift so median == target_med
if force_skin && isfinite(med_used)
    shift = target_med - med_used;
    Temp_C = Temp_C + shift;
    med_used = med_used + shift;
    recentered_by = shift;
end
end

%% ================== PER-WINDOW FEATURES =====================
function R = featuresOneWindow(seg, fs)
% Guards
for nm = ["AI2","AI3","Temp_C"]
    if ~ismember(nm, seg.Properties.VariableNames), seg.(nm) = nan(height(seg),1); end
end

ecg  = seg.AI2(:);
eda  = seg.AI3(:);
tmpC = seg.Temp_C(:);

% ECG HRV (time-domain; LF/HF only if >=60s)
[HR_mean, SDNN, RMSSD, NN50, pNN50, LF_power, HF_power, LF_HF, Total_power] = ...
    hrvFromSignalShort(ecg, fs, height(seg)/fs);

% EDA
x = eda - medianSafe(eda); dx = diff(x);
thrEDA = 0.5*stdSafe(dx); if ~isfinite(thrEDA) || thrEDA==0, thrEDA = 0; end
[~, locsEDA] = findpeaks(dx, 'MinPeakHeight', thrEDA, ...
    'MinPeakDistance', max(1, round(0.5*fs)));
dur_min  = height(seg)/fs/60;
SCR_rate = numel(locsEDA) / max(dur_min, eps);

% Temp: smoothing + robust slope
tmin = (0:height(seg)-1)'/fs/60;
tmpFilled = smoothTemp(fillTemp(tmpC), fs, 2.0);
[slopeC, meanC, ~, ~] = robustTempStats(tmin, tmpFilled);

% Assemble
R.HR_mean = HR_mean; R.SDNN = SDNN; R.RMSSD = RMSSD; R.NN50 = NN50; R.pNN50 = pNN50;
R.LF_power = LF_power; R.HF_power = HF_power; R.LF_HF = LF_HF; R.Total_power = Total_power;

R.EDA_mean = meanSafe(eda); R.EDA_std = stdSafe(eda); R.SCR_rate = SCR_rate;

R.Temp_mean_C          = meanC;
R.Temp_slope_C_per_min = slopeC;

R.ECG_p25 = prctileSafe(ecg, 25); R.ECG_p50 = prctileSafe(ecg, 50); R.ECG_p75 = prctileSafe(ecg, 75);
R.EDA_p25 = prctileSafe(eda, 25); R.EDA_p50 = prctileSafe(eda, 50); R.EDA_p75 = prctileSafe(eda, 75);
R.Temp_p25_C = prctileSafe(tmpC, 25); R.Temp_p50_C = prctileSafe(tmpC, 50); R.Temp_p75_C = prctileSafe(tmpC, 75);
end

%% ============= PER-VIDEO HRV (WHOLE SIGNAL) =================
function RV = hrvFromWholeSignal(ecg, fs)
ecg = double(ecg(:));
try
    [b,a] = butter(4, [5 20]/(fs/2), 'bandpass'); ecg_f = filtfilt(b,a, ecg);
catch
    ecg_f = ecg;
end

prom   = 0.5*stdSafe(ecg_f);
mndist = max(1, round(0.25*fs));
[~, locs] = findpeaks(ecg_f, 'MinPeakProminence', prom, 'MinPeakDistance', mndist);
if numel(locs) < 2
    [~, locs] = findpeaks(-ecg_f, 'MinPeakProminence', prom, 'MinPeakDistance', mndist);
end
if numel(locs) < 2
    RV = nanVideoHRVRow(); return;
end

rp_times = locs(:)/fs;
rr       = diff(rp_times);
r_times  = rp_times(2:end);

mask    = isfinite(rr) & rr > 0.3 & rr < 2.0;
rr      = rr(mask);
r_times = r_times(mask);
if numel(rr) < 2, RV = nanVideoHRVRow(); return; end

good    = isfinite(r_times) & isfinite(rr);
r_times = r_times(good); rr = rr(good);
[r_times, ia] = unique(r_times, 'stable'); rr = rr(ia);

HR_mean = 60 / meanSafe(rr);
SDNN    = stdSafe(rr);
RMSSD   = sqrt(meanSafe(diff(rr).^2));
NN50    = sum(abs(diff(rr))*1000 > 50);
pNN50   = 100 * NN50 / max(numel(rr)-1, 1);

LF_power = NaN; HF_power = NaN; LF_HF = NaN; Total_power = NaN;
dur_s = r_times(end) - r_times(1);
if numel(r_times) >= 3 && dur_s > 1
    rr_ms = (rr - meanSafe(rr)) * 1000;
    fsr   = 4;
    t_reg = r_times(1):1/fsr:r_times(end);
    if numel(t_reg) >= 64
        try
            rr_reg = interp1(r_times, rr_ms, t_reg, 'pchip', 'extrap');
            rr_reg = detrend(rr_reg, 1);
            nwin   = min(512, numel(rr_reg)); if nwin < 64, nwin = numel(rr_reg); end
            [Pxx, F] = pwelch(rr_reg, hamming(nwin), floor(0.5*nwin), [], fsr);
            LF_idx = F >= 0.04 & F < 0.15;
            HF_idx = F >= 0.15 & F <= 0.40;
            LF_power    = trapz(F(LF_idx), Pxx(LF_idx));
            HF_power    = trapz(F(HF_idx), Pxx(HF_idx));
            Total_power = trapz(F, Pxx);
            LF_HF       = LF_power / max(HF_power, eps);
        catch
        end
    end
end

RV.HR_mean         = HR_mean;
RV.SDNN            = SDNN;
RV.RMSSD           = RMSSD;
RV.NN50            = NN50;
RV.pNN50           = pNN50;
RV.LF_power        = LF_power;
RV.HF_power        = HF_power;
RV.LF_HF           = LF_HF;
RV.Total_power     = Total_power;
RV.RR_count        = numel(rr);
RV.Rpeak_count     = numel(locs);
RV.valid_rr_ratio  = numel(rr) / max(numel(diff(rp_times)), 1);
end

%% ================== SHORT HRV (WINDOW) ======================
function [HR_mean, SDNN, RMSSD, NN50, pNN50, LF_power, HF_power, LF_HF, Total_power] = ...
         hrvFromSignalShort(ecg, fs, seg_dur_s)
ecg = double(ecg(:));
try
    [b,a] = butter(4, [5 20]/(fs/2), 'bandpass'); ecg_f = filtfilt(b,a, ecg);
catch
    ecg_f = ecg;
end

prom      = 0.5*stdSafe(ecg_f);
mndist    = max(1, round(0.25*fs));
[~, locs] = findpeaks(ecg_f, 'MinPeakProminence', prom, 'MinPeakDistance', mndist);
if numel(locs) < 2
    [~, locs] = findpeaks(-ecg_f, 'MinPeakProminence', prom, 'MinPeakDistance', mndist);
end
if numel(locs) < 2
    HR_mean = NaN; SDNN = NaN; RMSSD = NaN; NN50 = NaN; pNN50 = NaN;
    LF_power = NaN; HF_power = NaN; LF_HF = NaN; Total_power = NaN; return;
end

rp_times = locs(:)/fs;
rr       = diff(rp_times);
r_times  = rp_times(2:end);

mask    = isfinite(rr) & rr > 0.3 & rr < 2.0;
rr      = rr(mask);
r_times = r_times(mask);
if numel(rr) < 2
    HR_mean = NaN; SDNN = NaN; RMSSD = NaN; NN50 = NaN; pNN50 = NaN;
    LF_power = NaN; HF_power = NaN; LF_HF = NaN; Total_power = NaN; return;
end

HR_mean = 60 / meanSafe(rr);
SDNN    = stdSafe(rr);
RMSSD   = sqrt(meanSafe(diff(rr).^2));
NN50    = sum(abs(diff(rr))*1000 > 50);
pNN50   = 100 * NN50 / max(numel(rr)-1,1);

LF_power = NaN; HF_power = NaN; LF_HF = NaN; Total_power = NaN;
if seg_dur_s >= 60 && numel(r_times) >= 3 && (r_times(end)-r_times(1)) > 1.0
    rr_ms = (rr - meanSafe(rr)) * 1000;
    fsr   = 4;
    t_reg = r_times(1):1/fsr:r_times(end);
    if numel(t_reg) >= 32
        try
            rr_reg = interp1(r_times, rr_ms, t_reg, 'pchip', 'extrap');
            rr_reg = detrend(rr_reg, 1);
            nwin = min(256, numel(rr_reg)); if nwin < 32, nwin = numel(rr_reg); end
            [Pxx,F] = pwelch(rr_reg, hamming(nwin), floor(0.5*nwin), [], fsr);
            LF_idx = F >= 0.04 & F < 0.15;
            HF_idx = F >= 0.15 & F <= 0.40;
            LF_power    = trapz(F(LF_idx), Pxx(LF_idx));
            HF_power    = trapz(F(HF_idx), Pxx(HF_idx));
            Total_power = trapz(F, Pxx);
            LF_HF       = LF_power / max(HF_power, eps);
        catch
        end
    end
end
end

%% ===================== ROW TEMPLATES ========================
function R = nanFeatureRow()
names = {'HR_mean','SDNN','RMSSD','NN50','pNN50', ...
         'LF_power','HF_power','LF_HF','Total_power', ...
         'EDA_mean','EDA_std','SCR_rate', ...
         'Temp_mean_C','Temp_slope_C_per_min', ...
         'ECG_p25','ECG_p50','ECG_p75', ...
         'EDA_p25','EDA_p50','EDA_p75', ...
         'Temp_p25_C','Temp_p50_C','Temp_p75_C', ...
         'subject','video_index','emotion','class_label','win_start_s','win_end_s'};
for k=1:numel(names)
    if ismember(names{k}, {'subject','emotion','class_label'})
        R.(names{k}) = "";
    elseif ismember(names{k}, {'video_index'})
        R.(names{k}) = NaN;
    else
        R.(names{k}) = NaN;
    end
end
end

function RV = nanVideoHRVRow()
names = {'HR_mean','SDNN','RMSSD','NN50','pNN50', ...
         'LF_power','HF_power','LF_HF','Total_power', ...
         'RR_count','Rpeak_count','valid_rr_ratio', ...
         'subject','video_index','emotion','class_label','duration_s', ...
         'Temp_mean_C','Temp_slope_C_per_min','Temp_median_C','Temp_iqr_C', ...
         'Temp_mode','Temp_vref_V'};
for k=1:numel(names)
    if ismember(names{k}, {'subject','emotion','class_label','Temp_mode'})
        RV.(names{k}) = "";
    elseif ismember(names{k}, {'video_index'})
        RV.(names{k}) = NaN;
    else
        RV.(names{k}) = NaN;
    end
end
end

%% ======== Utility: robust numeric/string/datetime handling ========
function x = toNumeric(x)
if isnumeric(x), x = double(x); return; end
s = toStringVector(x);
vals = str2double(s);
if all(isnan(vals) & s ~= "")
    dt = tryParseDatetime(s);
    if any(~isnat(dt))
        t0 = dt(find(~isnat(dt),1,'first'));
        vals = seconds(dt - t0);
    end
end
if all(isnan(vals)), x = nan(numel(s),1); else, x = vals; end
end

function s = toStringVector(col)
if isstring(col)
    s = col;
elseif iscell(col)
    try s = string(col); catch
        s = strings(size(col));
        for ii=1:numel(col)
            if ischar(col{ii}) || isstring(col{ii}), s(ii) = string(col{ii});
            elseif isnumeric(col{ii}) && isscalar(col{ii}), s(ii) = string(col{ii});
            else, s(ii) = ""; end
        end
    end
elseif ischar(col)
    s = string(col);
else
    s = string(col);
end
end

function dt = tryParseDatetime(s)
dt = NaT(size(s));
fmts = {'yyyy-MM-dd''T''HH:mm:ss.SSS','yyyy-MM-dd''T''HH:mm:ss', ...
        'dd/MM/yyyy HH:mm:ss','MM/dd/yyyy HH:mm:ss', ...
        'yyyy-MM-dd HH:mm:ss','HH:mm:ss.SSS','HH:mm:ss'};
for k = 1:numel(fmts)
    try, cand = datetime(s, 'InputFormat', fmts{k}, 'TimeZone','local'); catch, cand = NaT(size(s)); end
    fillIdx = isnat(dt) & ~isnat(cand);
    dt(fillIdx) = cand(fillIdx);
    if all(~isnat(dt)), break; end
end
end

%% ======== NaN-safe stats & temp helpers =========
function m = meanSafe(x), x = x(isfinite(x)); if isempty(x), m = NaN; else, m = mean(x); end, end
function s = stdSafe(x),  x = x(isfinite(x)); if isempty(x), s = NaN; else, s = std(x);  end, end
function p = prctileSafe(x,q), x=x(isfinite(x)); if isempty(x), p=NaN; else, p=prctile(x,q); end, end
function med = medianSafe(x), x = x(isfinite(x)); if isempty(x), med = NaN; else, med = median(x); end, end

function x = fillTemp(x)
x = double(x(:));
try
    x = fillmissing(x,'linear','EndValues','nearest');
catch
    nanidx = isnan(x);
    if any(~nanidx)
        first = find(~nanidx,1,'first'); last = find(~nanidx,1,'last');
        x(1:first-1) = x(first); x(last+1:end) = x(last);
        for ii = first+1:last-1
            if isnan(x(ii)), x(ii) = (x(ii-1)+x(ii+1))/2; end
        end
    end
end
end

function x = smoothTemp(x, fs, smooth_sec)
w = max(1, round(smooth_sec*fs));
x = movmedian(x, w, 'omitnan');
end

function [slope, meanC, medC, iqrC] = robustTempStats(tmin, tempC)
meanC = mean(tempC,'omitnan');
medC  = median(tempC,'omitnan');
q25   = prctile(tempC,25); q75 = prctile(tempC,75);
iqrC  = q75 - q25;
slope = NaN;
try
    if exist('robustfit','file')
        b = robustfit(tmin, double(tempC));  % b0 + b1*t
        slope = b(2);
    else
        A = [tmin, ones(size(tmin))];
        coeff = A \ double(tempC);
        slope = coeff(1);
    end
catch
end
end

%% ======== Names =========
function out = slugify(s)
s = string(s);
out = lower(regexprep(s, '[^A-Za-z0-9]+', '_'));
out = regexprep(out,'(^_+|_+$)','');
out = extractBefore(out + "______________________________", 32);
end

function sh = sheetname(s)
sh = char(s);
sh = regexprep(sh, '[:\\/\?\*\[\]]', '_');
if numel(sh) > 31, sh = sh(1:31); end
if isempty(sh), sh = 'Sheet'; end
sh = string(sh);
end

function c = emotionalClass(emo)
if strcmpi(string(emo), "Neutral"), c = "Neutral"; else, c = "Emotional"; end
end
