function sample16_brd_alpha_sweep(decay_csv, output_dir, brd_dir, alpha_values, normalize_first_echo)
% Run the T2process MATLAB brd1.p solver on Sample 16 experimental decay.
addpath(brd_dir);
assert(exist('brd1', 'file') ~= 0 && exist('AmplitudeSample', 'file') ~= 0, 'T2process MATLAB BRD files missing');
raw = readmatrix(decay_csv);
valid = isfinite(raw(:,1)) & isfinite(raw(:,2)) & raw(:,1) > 0;
time_ms = raw(valid,1);
signal = raw(valid,2);
[time_ms, order] = sort(time_ms);
signal = signal(order);
assert(numel(time_ms) >= 200, 'Insufficient experimental echoes');
if nargin >= 5 && normalize_first_echo
    signal = signal / signal(1);
end

% T2process main_1D_T2process.m uses AmplitudeSample then exp(-t/T2) and brd1.
[pick_t, pick_y] = AmplitudeSample(time_ms, signal, 200, 0);
pick_t = pick_t(:);
pick_y = pick_y(:);
assert(numel(unique(pick_t)) == numel(pick_t), 'Repeated sampled echo times');
t2_ms = logspace(-2, 5, 200).';  % T2process batch/run_BRD.m grid
kernel = exp(-pick_t * (t2_ms'.^(-1)));
if nargin < 4
    alpha = logspace(0, 5, 11).';
else
    alpha = alpha_values(:);
end
assert(all(isfinite(alpha) & alpha > 0), 'Alpha must be positive and finite');
spectra = zeros(numel(t2_ms), numel(alpha));
summary = zeros(numel(alpha), 4);
for i = 1:numel(alpha)
    spectrum = brd1(kernel, pick_y, alpha(i));
    spectrum = spectrum(:);
    assert(numel(spectrum) == numel(t2_ms) && all(isfinite(spectrum)), 'Invalid BRD spectrum');
    spectra(:,i) = spectrum;
    fitted = kernel * spectrum;
    [~, peak_index] = max(spectrum);
    summary(i,:) = [alpha(i), t2_ms(peak_index), norm(pick_y - fitted) / norm(pick_y), min(spectrum)];
    fprintf('alpha=%g peak_T2_ms=%g relative_residual=%g min_amplitude=%g\n', summary(i,:));
end

if ~isfolder(output_dir), mkdir(output_dir); end
writematrix([t2_ms, spectra], fullfile(output_dir, 'brd_t2_spectra.csv'));
writematrix(summary, fullfile(output_dir, 'brd_summary.csv'));
writematrix([pick_t, pick_y], fullfile(output_dir, 'brd_sampled_decay.csv'));
save(fullfile(output_dir, 'brd_results.mat'), 'alpha', 't2_ms', 'spectra', 'summary', 'pick_t', 'pick_y');
end
