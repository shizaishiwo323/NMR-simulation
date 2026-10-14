function ideal_two_pore_brd_batch(input_dir, output_dir, brd_dir, alpha)
% Run the T2process MATLAB brd1.p solver at one fixed alpha on every decay CSV in input_dir.
% Same steps as sample16_brd_alpha_sweep.m: first-echo normalization, AmplitudeSample (200 points),
% T2 grid logspace(-2, 5, 200) ms, kernel exp(-t/T2), brd1(kernel, y, alpha).
% Each input CSV: column 1 = time (ms, > 0), column 2 = signal. Writes <name>_brd.csv with
% [T2_ms, amplitude] and <name>_sampled.csv with [t_ms, y, fitted].
addpath(brd_dir);
assert(exist('brd1', 'file') ~= 0 && exist('AmplitudeSample', 'file') ~= 0, 'T2process MATLAB BRD files missing');
assert(isfinite(alpha) && alpha > 0, 'Alpha must be positive and finite');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
files = dir(fullfile(input_dir, '*.csv'));
assert(~isempty(files), 'No decay CSV files found');
t2_ms = logspace(-2, 5, 200).';
for k = 1:numel(files)
    raw = readmatrix(fullfile(input_dir, files(k).name));
    valid = isfinite(raw(:,1)) & isfinite(raw(:,2)) & raw(:,1) > 0;
    [time_ms, order] = sort(raw(valid,1));
    signal = raw(valid,2);
    signal = signal(order) / signal(order(1));
    [pick_t, pick_y] = AmplitudeSample(time_ms, signal, 200, 0);
    pick_t = pick_t(:);
    pick_y = pick_y(:);
    kernel = exp(-pick_t * (t2_ms'.^(-1)));
    spectrum = brd1(kernel, pick_y, alpha);
    spectrum = spectrum(:);
    assert(numel(spectrum) == numel(t2_ms) && all(isfinite(spectrum)), 'Invalid BRD spectrum');
    [~, stem] = fileparts(files(k).name);
    writematrix([t2_ms, spectrum], fullfile(output_dir, [stem '_brd.csv']));
    writematrix([pick_t, pick_y, kernel * spectrum], fullfile(output_dir, [stem '_sampled.csv']));
    fprintf('%s alpha=%g relative_residual=%g\n', stem, alpha, norm(pick_y - kernel * spectrum) / norm(pick_y));
end
end
