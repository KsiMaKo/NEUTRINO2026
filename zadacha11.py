import geant4_pybind as g4
import numpy as np
import numpy.random as rnd

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

        # Сцинтиллятор C9H12
        scint = g4.G4Material("Tsyntsylator", 0.876 * g4.g / g4.cm3, 2)
        scint.AddElement(nist.FindOrBuildElement("C"), 9)
        scint.AddElement(nist.FindOrBuildElement("H"), 12)

        prpt_scint = g4.G4MaterialPropertiesTable()
        prpt_scint.AddProperty("RINDEX", E_opt, g4.G4doubleVector([1.5, 1.5]))
        prpt_scint.AddProperty("ABSLENGTH", E_opt, g4.G4doubleVector([8.0 * g4.m, 8.0 * g4.m]))
        scint.SetMaterialPropertiesTable(prpt_scint)


        glass = nist.FindOrBuildMaterial("G4_Pyrex_Glass")
        prpt_glass = g4.G4MaterialPropertiesTable()
        prpt_glass.AddProperty("RINDEX", E_opt, g4.G4doubleVector([1.5, 1.5]))
        glass.SetMaterialPropertiesTable(prpt_glass)

        world_mat = nist.FindOrBuildMaterial("G4_AIR")

        world_box = g4.G4Box("World", 3 * g4.m, 3 * g4.m, 3 * g4.m)
        world_log = g4.G4LogicalVolume(world_box, world_mat, "World")
        world_phys = g4.G4PVPlacement(None, g4.G4ThreeVector(), world_log, "World", None, False, 0)

        sphere = g4.G4Orb("Sphere", self.R)
        sphere_vol = g4.G4LogicalVolume(sphere, scint, "Sphere")
        g4.G4PVPlacement(None, g4.G4ThreeVector(), sphere_vol, "Sphere", world_log, False, 0)

        # Цилиндры фотомножителей
        fmt_solid = g4.G4Tubs("FMT", 0, self.r_fmt, 0.01 * g4.m, 0, 360 * g4.deg)
        fmt_vol = g4.G4LogicalVolume(fmt_solid, glass, "FMT")

        for k in range(40):
            zk = -self.R + 2.0 * self.R * (k + 0.5) / 40
            rk = np.sqrt(self.R**2 - zk**2)
            phi_k = k * np.pi / np.e

            pos = np.array([rk * np.cos(phi_k), rk * np.sin(phi_k), zk])
            unit_p = pos / self.R

            # Поверну цилиндры фмт к центру
            rot = g4.G4RotationMatrix()
            rot.rotateY(float(np.arccos(unit_p[2])))
            rot.rotateZ(float(np.arctan2(unit_p[1], unit_p[0])))
            self.rotations.append(rot)

            rot.invert()
            p_vec = g4.G4ThreeVector(*(pos * 0.99))  # чуть внутри сферы, чтобы фмт поместились
            g4.G4PVPlacement(rot, p_vec, fmt_vol, f"FMT_{k}", sphere_vol, False, k)

        return world_phys



class Generator(g4.G4VUserPrimaryGeneratorAction):

    def __init__(self):
        super().__init__()
        self.gun = g4.G4ParticleGun(1)
        particle = g4.G4ParticleTable.GetParticleTable().FindParticle("opticalphoton")
        self.gun.SetParticleDefinition(particle)
        self.gun.SetParticleEnergy(3.0 * g4.eV)
        self.gun.SetParticlePosition(g4.G4ThreeVector(0, 0, 0))

    def GeneratePrimaries(self, event):
        costh = rnd.uniform(-1, 1)
        phi = rnd.uniform(0, 2 * np.pi)
        sinth = np.sqrt(1 - costh**2)
        direction = g4.G4ThreeVector(sinth * np.cos(phi), sinth * np.sin(phi), costh)
        self.gun.SetParticleMomentumDirection(direction)

        # Задаём поляризацию
        pol = direction.cross(
            g4.G4ThreeVector(0, 0, 1)
            if abs(costh) < 0.9
            else g4.G4ThreeVector(1, 0, 0))
        self.gun.SetParticlePolarization(pol.unit())

        self.gun.GeneratePrimaryVertex(event)


class Stepping(g4.G4UserSteppingAction):

    def __init__(self):
        super().__init__()
        self.hits = np.zeros(40, dtype=int)

    def UserSteppingAction(self, step):
        trk = step.GetTrack()
        vol = trk.GetVolume() #Текущий объем трека

        if vol and "FMT" in vol.GetName():
            self.hits[vol.GetCopyNo()] += 1
            trk.SetTrackStatus(g4.G4TrackStatus.fStopAndKill) #поглощаем 

run_mgr = g4.G4RunManager()
run_mgr.SetUserInitialization(Detector())

physics = g4.FTFP_BERT()
physics.RegisterPhysics(g4.G4OpticalPhysics())
run_mgr.SetUserInitialization(physics)

gen = Generator()
step = Stepping()
run_mgr.SetUserAction(gen)
run_mgr.SetUserAction(step)

run_mgr.Initialize()

N_photons = 10000
run_mgr.BeamOn(N_photons)

photons_per_fmt = rnd.poisson(0.5 * step.hits)
total_charge = np.sum(photons_per_fmt)

exp_charge = N_photons * (40 * 0.01 / 16.0) * np.exp(-2.0 / 8.0) * 0.5

print(f"Попало фотонов: {np.sum(step.hits)}")
print(f"Суммарный заряд: {total_charge}")
print(f"Теоретически ожидалось: {exp_charge:.1f}")
