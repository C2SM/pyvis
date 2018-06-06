# What's New

## Versions

### 1.1.0 (unreleased)

 * updated installation files (closes [GH37][i37])
 * use `np.asarray` instead of `np.array`, because it does not copy the data (closes [GH2][i2])
 * Update 0
   * improved the exercises (more explanations, removed unnecessary parts, slight restructuring)
   * added disclaimer when using a system command, e.g. `! head file` (closes [GH34][i34])
   * pandas as last exercise (closes [GH4][i4])
   * ex0_1: mention shift + tab to get to the help (closes [GH15][i15])
   * ex0_2: `np.random.randint` is exclusive (closes [GH1][i1])
   * ex0_2: Add more exercises on `np.arange` (closes [GH39][i39])

 * Updated Part 1
   * corrected some typos
   * restructured some exercises
   * explain arguments of `plt.subplots` (closes [GH6][i6] and [GH7][i7])
   * explain why np.newaxis is needed in ex1_2 (broadcasting) (closes [GH10][i10])
   * replaced plt.scatter by ax.scatter in ex1_3 (closes [GH12][i12])
   * now plots correct data in last exercise of ex1_4 (closes [GH13][i13])
   * hexbin can handle NaNs in ex1_5 (closes [GH14][i14])
 * Updated Part 2
   * typos, small changes to text
   * simplify ex2_2
 * Updated Part 3
   * use `mplotutils as mpu` instead of `utils`
   * simplifications & typos
   * ex3_1 replaced the color 'teal' with 'green' (closes [GH11][i11])
   * ex3_1 use `plt.scatter?` for help, as `ax.scatter?` returns the cartopy scatter function (closes [GH18][i18])
   * ex3_2 fix typos and colormap range (closes [GH17][i17] and [GH19][i19])
   * ex3_2 explain the 351 (and not 350) in `np.arange(-10, 351, 20)` (closes [GH22][i22])
   * ex3_5 typo (closes [GH20][i20])
   * ex3_6 correct longitude range (closes [GH21][i21])
   * ex3_8 rephrase: we cannot do along-line color changes (closes [GH24][i24])
 * Updated Part 4
   * use `mplotutils as mpu` instead of `utils` for ex4_1 and ex4_2
   * correct typos

### 1.0.0 (06.02.2018)

 * Version used for the first pyvis course.
 * Create release.



[i1]: https://github.com/C2SM/pyvis/issues/1
[i2]: https://github.com/C2SM/pyvis/issues/2
[i4]: https://github.com/C2SM/pyvis/issues/4
[i6]: https://github.com/C2SM/pyvis/issues/6
[i7]: https://github.com/C2SM/pyvis/issues/7
[i10]: https://github.com/C2SM/pyvis/issues/10
[i11]: https://github.com/C2SM/pyvis/issues/11
[i12]: https://github.com/C2SM/pyvis/issues/12
[i13]: https://github.com/C2SM/pyvis/issues/13
[i14]: https://github.com/C2SM/pyvis/issues/14
[i15]: https://github.com/C2SM/pyvis/issues/15
[i17]: https://github.com/C2SM/pyvis/issues/17
[i19]: https://github.com/C2SM/pyvis/issues/19
[i20]: https://github.com/C2SM/pyvis/issues/20
[i21]: https://github.com/C2SM/pyvis/issues/21
[i22]: https://github.com/C2SM/pyvis/issues/22
[i24]: https://github.com/C2SM/pyvis/issues/24
[i34]: https://github.com/C2SM/pyvis/issues/34
[i37]: https://github.com/C2SM/pyvis/issues/37
[i39]: https://github.com/C2SM/pyvis/issues/39
