import glob

urfd_path = '/home/anunez/Downloads/URFD_OF/'
multicam_path = '/home/anunez/Downloads/Multicam_OF/'
fdd_path = '/home/anunez/Downloads/FDD_OF/'
L = 10

content = [
    ['URFD', urfd_path],
    ['FDD', fdd_path],
    ['Multicam', multicam_path]
]

# URFD, FDD
for dataset_name, dataset_path in content[:2]:
    falls = len(glob.glob(dataset_path + 'Falls/*'))
    notfalls = len(glob.glob(dataset_path + 'NotFalls/*'))
    print('-'*10)
    print(dataset_name)
    print('-'*10)
    print('Fall sequences: {}, ADL sequences: {} (Total: {})'.format(
        falls, notfalls, falls + notfalls)
    )

    falls = glob.glob(dataset_path + 'Falls/*')
    fall_stacks = sum(
        [len(glob.glob(fall +'/*')) - L + 1 for fall in falls]
    )
    fall_frames = sum(
        [len(glob.glob(fall +'/*')) for fall in falls]
    )
    notfalls = glob.glob(dataset_path + 'NotFalls/*')
    nofall_stacks = sum(
        [len(glob.glob(notfall +'/*')) - L + 1 for notfall in notfalls])
    nofall_frames = sum(
        [len(glob.glob(notfall +'/*'))  for notfall in notfalls])

    print('Fall stacks: {}, ADL stacks: {} (Total: {})'.format(
        fall_stacks, nofall_stacks, fall_stacks + nofall_stacks)
    )
    print('Fall frames: {}, ADL frames: {} (Total: {})\n\n'.format(
        fall_frames, nofall_frames,fall_frames + nofall_frames)
    )
    
# Multicam
dataset_name, dataset_path = content[2]
print('-'*10)
print(dataset_name)
print('-'*10)
scenarios = glob.glob(dataset_path + '*')
nb_falls, nb_notfalls = 0, 0
nb_fall_stacks, nb_notfall_stacks = 0, 0
nb_fall_frames, nb_notfall_frames = 0, 0
for scenario in scenarios:
    nb_falls += len(glob.glob(scenario + '/Falls/*'))
    nb_notfalls += len(glob.glob(scenario + '/NotFalls/*'))

    falls = glob.glob(scenario + '/Falls/*')
    nb_fall_stacks += sum(
        [len(glob.glob(fall +'/*')) - L + 1 for fall in falls]
    )
    nb_fall_frames += sum(
        [len(glob.glob(fall +'/*')) for fall in falls]
    )
    notfalls = glob.glob(scenario + '/NotFalls/*')
    nb_notfall_stacks += sum(
        [len(glob.glob(notfall +'/*')) - L + 1 for notfall in notfalls])
    nb_notfall_frames += sum(
        [len(glob.glob(notfall +'/*')) for notfall in notfalls])

print('Fall sequences: {}, ADL sequences: {} (Total: {})'.format(
    nb_falls, nb_notfalls, nb_falls + nb_notfalls)
)

print('Fall stacks: {}, ADL stacks: {} (Total: {})'.format(
    nb_fall_stacks, nb_notfall_stacks, nb_fall_stacks + nb_notfall_stacks)
)

print('Fall frames: {}, ADL frames: {} (Total: {})\n\n'.format(
    nb_fall_frames, nb_notfall_frames, nb_fall_frames + nb_notfall_frames)
)