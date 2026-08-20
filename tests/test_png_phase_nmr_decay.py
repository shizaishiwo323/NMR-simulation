from advanced_tools import png_phase_nmr_decay as mod
from scipy.sparse import csr_matrix


def test_save_pygimli_mesh_failure_is_nonfatal(monkeypatch, tmp_path):
    class BrokenMesh:
        def save(self, path):
            raise RuntimeError("cannot save here")

    monkeypatch.setattr(mod, "pg", object())

    assert mod.save_pygimli_mesh({"pygimli_mesh": BrokenMesh()}, tmp_path / "mesh.bms") is False


def test_triangular_solver_adds_jitter_for_singular_factorization(monkeypatch):
    calls = {"n": 0}

    def fake_factorized(matrix):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Factor is exactly singular")
        return lambda rhs: rhs

    monkeypatch.setattr(mod, "assemble_triangular_operator", lambda labels, params: ({}, csr_matrix([[0.0]]), mod.np.array([1.0]), mod.np.array([0.0]), mod.np.array([0.0]), {}))
    monkeypatch.setattr(mod, "triangle_quality_table", lambda mesh: mod.pd.DataFrame())
    monkeypatch.setattr(mod, "summarize_mesh_quality", lambda mesh, quality: {})
    monkeypatch.setattr(mod, "factorized", fake_factorized)
    params = mod.SimulationParams(1, 1, 2, 3000, 0.005, 0, 1, 0, 32, "triangular", 48, 24, 3000)

    _, _, stats = mod.solve_decay_triangular(mod.np.zeros((1, 1), dtype=int), params)

    assert calls["n"] == 2
    assert stats["triangular_factorization_jitter"] > 0
