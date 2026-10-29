import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
import geant4_pybind as g4

rnd.seed(42)

class Detector(g4.G4VUserDetectorConstruction):
    def __init__(self):
        super().__init__()
        self.R = 2.0 * g4.m
        self.r_fmt = 0.1 * g4.m
        self.rotations = []

    def Construct(self):
        nist = g4.G4NistManager.Instance()
        E_opt = g4.G4doubleVector([2.0 * g4.eV, 4.0 * g4.eV])

        scint = g4.G4Material("Tsyntsylator", 0.876 * g4.g / g4.cm3, 2)
        scint.AddElement(nist.FindOrBuildElement("C"), 9)
        scint.AddElement(nist.FindOrBuildElement("H"), 12)

        prpt_scint = g4.G4MaterialPropertiesTable()
        prpt_scint.AddProperty("RINDEX", E_opt, g4.G4doubleVector([1.5, 1.5]))
        prpt_scint.AddProperty("ABSLENGTH", E_opt, g4.G4doubleVector([8.0 * g4.m, 8.0 * g4.m]))

        # Спектр высвечивания
        e_grid = [float(x) for x in np.linspace(2.0, 4.0, 21) * g4.eV]
        spec_vals = [float(np.exp(-0.5 * ((x - 3.0 * g4.eV) / (0.25 * g4.eV))**2)) for x in e_grid]
        prpt_scint.AddProperty("SCINTILLATIONCOMPONENT1", g4.G4doubleVector(e_grid), g4.G4doubleVector(spec_vals))

        # Пропертиз сцинтилляции
        prpt_scint.AddConstProperty("SCINTILLATIONYIELD", 10000.0 / g4.MeV)
        prpt_scint.AddConstProperty("SCINTILLATIONTIMECONSTANT1", 10.0 * g4.ns)
        prpt_scint.AddConstProperty("RESOLUTIONSCALE", 1.0)
        scint.SetMaterialPropertiesTable(prpt_scint)
      
        glass = nist.FindOrBuildMaterial("G4_Pyrex_Glass")
        prpt_glass = g4.G4MaterialPropertiesTable()
        prpt_glass.AddProperty("RINDEX", E_opt, g4.G4doubleVector([1.5, 1.5]))
        glass.SetMaterialPropertiesTable(prpt_glass)
        world_mat = nist.FindOrBuildMaterial("G4_AIR")

        world_box = g4.G4Box("World", 3 * g4.m, 3 * g4.m, 3 * g4.m)
        world_vol = g4.G4LogicalVolume(world_box, world_mat, "World")
        world_phys = g4.G4PVPlacement(None, g4.G4ThreeVector(), world_vol, "World", None, False, 0)

        sphere = g4.G4Orb("Sphere", self.R)
        sphere_vol = g4.G4LogicalVolume(sphere, scint, "Sphere")
        g4.G4PVPlacement(None, g4.G4ThreeVector(), sphere_vol, "Sphere", world_vol, False, 0)

        fmt_solid = g4.G4Tubs("FMT", 0, self.r_fmt, 0.01 * g4.m, 0, 360 * g4.deg)
        fmt_vol = g4.G4LogicalVolume(fmt_solid, glass, "FMT")

        for k in range(40):
            zk = -self.R + 2.0 * self.R * (k + 0.5) / 40
            rk = np.sqrt(max(0.0, self.R**2 - zk**2))
            phi_k = k * np.pi / np.e

            pos = np.array([rk * np.cos(phi_k), rk * np.sin(phi_k), zk])
            unit_p = pos / self.R

            rot = g4.G4RotationMatrix()
            rot.rotateY(float(np.arccos(unit_p[2])))
            rot.rotateZ(float(np.arctan2(unit_p[1], unit_p[0])))
            self.rotations.append(rot)

            rot.invert()
            p_vec = g4.G4ThreeVector(*(pos * 0.99))
            g4.G4PVPlacement(rot, p_vec, fmt_vol, f"FMT_{k}", sphere_vol, False, k)

        return world_phys


class ElectronGenerator(g4.G4VUserPrimaryGeneratorAction):
    def __init__(self):
        super().__init__()
        self.gun = g4.G4ParticleGun(1)
        particle = g4.G4ParticleTable.GetParticleTable().FindParticle("e-")
        self.gun.SetParticleDefinition(particle)
        self.gun.SetParticleEnergy(2.0 * g4.MeV)
        self.gun.SetParticlePosition(g4.G4ThreeVector(0, 0, 0))

    def GeneratePrimaries(self, event):
      # Из центра стреляю
        costh = rnd.uniform(-1, 1)
        phi = rnd.uniform(0, 2 * np.pi)
        sinth = np.sqrt(1 - costh**2)
        direction = g4.G4ThreeVector(sinth * np.cos(phi), sinth * np.sin(phi), costh)
        self.gun.SetParticleMomentumDirection(direction)
        self.gun.GeneratePrimaryVertex(event)


class Stepping(g4.G4UserSteppingAction):
    def __init__(self):
        super().__init__()
        self.hits = np.zeros(40, dtype=int)

    def UserSteppingAction(self, step):
        trk = step.GetTrack()
        if trk.GetDefinition().GetParticleName() == "opticalphoton":
            vol = trk.GetVolume()
            if vol and "FMT" in vol.GetName():
                self.hits[vol.GetCopyNo()] += 1
                trk.SetTrackStatus(g4.G4TrackStatus.fStopAndKill)


run_mgr = g4.G4RunManager()
run_mgr.SetUserInitialization(Detector())

physics = g4.FTFP_BERT()
physics.RegisterPhysics(g4.G4OpticalPhysics())
run_mgr.SetUserInitialization(physics)

step = Stepping()
run_mgr.SetUserAction(ElectronGenerator())
run_mgr.SetUserAction(step)
run_mgr.Initialize()

# Симулякр запускается
N_events = 20
event_charges = []

for ev in range(N_events):
    step.hits.fill(0)
    run_mgr.BeamOn(1)

    photons = rnd.poisson(0.5 * step.hits)
    total_q = np.sum(photons)
    event_charges.append(total_q)
    print(f"Событие {ev+1}: хитов = {np.sum(step.hits)}, заряд Q = {total_q}")

charges = np.array(event_charges)

mean_Q = np.mean(charges)
std_Q = np.std(charges, ddof=1)


Q_theo_zat = (2.0 * 10000) * 0.025 * np.exp(-2.0 / 8.0) * 0.5
Q_theo_no_zat = (2.0 * 10000) * 0.025 * 0.5

print(f"без затухания: {Q_theo_no_zat:.1f}")
print(f"С затуханием:   {Q_theo_zat:.1f}")
print(f"Geant4: {mean_Q:.1f} ± {std_Q:.1f}")
