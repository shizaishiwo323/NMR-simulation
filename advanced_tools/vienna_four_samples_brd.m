function vienna_four_samples_brd(data_root, output_dir, brd_dir)
% Run the T2process brd1.p solver for four Vienna carbonate samples.
addpath(brd_dir);
assert(exist('brd1', 'file') ~= 0 && exist('AmplitudeSample', 'file') ~= 0, ...
    'T2process MATLAB BRD files missing');

sample_ids = ["24", "25", "26", "34"];
test_names = ["first", "second"];
alpha_values = [0.1, 0.5, 1.0];
t2_ms = logspace(-2, 5, 200).';
summary = table();
if ~isfolder(output_dir), mkdir(output_dir); end

for test_index = 1:numel(test_names)
    for sample_index = 1:numel(sample_ids)
        sample_id = sample_ids(sample_index);
        test_name = test_names(test_index);
        if test_name == "first"
            pea_path = fullfile(data_root, 'first test', sample_id + ".1.pea");
        else
            pea_path = fullfile(data_root, 'second test', "M14" + sample_id + ".1.pea");
        end
        raw = readmatrix(pea_path, 'FileType', 'text', 'NumHeaderLines', 4);
        valid = all(isfinite(raw(:, 1:2)), 2) & raw(:, 1) > 0;
        time_ms = raw(valid, 1);
        signal = raw(valid, 2);
        [time_ms, order] = sort(time_ms);
        signal = signal(order);
        [pick_time, pick_signal] = AmplitudeSample(time_ms, signal, 200, 0);
        pick_time = pick_time(:);
        pick_signal = pick_signal(:);
        kernel = exp(-pick_time * (t2_ms'.^(-1)));

        spectra = zeros(numel(t2_ms), numel(alpha_values));
        for alpha_index = 1:numel(alpha_values)
            alpha = alpha_values(alpha_index);
            spectrum = brd1(kernel, pick_signal, alpha);
            spectrum = spectrum(:);
            assert(numel(spectrum) == numel(t2_ms) && all(isfinite(spectrum)), ...
                'Invalid BRD spectrum for sample %s %s alpha %g', sample_id, test_name, alpha);
            fitted = kernel * spectrum;
            spectra(:, alpha_index) = spectrum;
            [~, peak_index] = max(spectrum);
            row = table(sample_id, test_name, alpha, t2_ms(peak_index), ...
                norm(pick_signal - fitted) / norm(pick_signal), min(spectrum), ...
                'VariableNames', {'sample_id', 'test', 'alpha', 'peak_t2_ms', ...
                'relative_residual', 'minimum_amplitude'});
            summary = [summary; row]; %#ok<AGROW>
        end

        spectrum_table = table(t2_ms, spectra(:, 1), spectra(:, 2), spectra(:, 3), ...
            'VariableNames', {'t2_ms', 'alpha_0p1', 'alpha_0p5', 'alpha_1'});
        writetable(spectrum_table, fullfile(output_dir, ...
            "sample" + sample_id + "_" + test_name + "_brd_spectra.csv"));
        writetable(table(pick_time, pick_signal), fullfile(output_dir, ...
            "sample" + sample_id + "_" + test_name + "_brd_sampled_decay.csv"));
    end
end

assert(height(summary) == 24, 'Expected 24 BRD results');
writetable(summary, fullfile(output_dir, 'brd_summary.csv'));
end
