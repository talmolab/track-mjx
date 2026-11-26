# DMPO prior checkpoints

Frozen SCAMPER prior checkpoints that the DMPO configs load as
`${oc.env:DMPO_PRIOR_ROOT}/<id>`. Each id is a directory holding the
pinned `PriorNetwork_<step>/` checkpoint (`config`, `policy`,
`train_state`, `_CHECKPOINT_METADATA`). The pinned step is the highest
step of its id, so configs that leave the step unset resolve to it.

Check a local copy against this file (exit status 0 means every file
matches):

    python tools/make_prior_manifest.py --verify --root "$DMPO_PRIOR_ROOT"

The directory digest is the sha256 of the lines
`<relative path>:<file sha256>` (one per file, sorted by path).

## Checkpoints

| id | step dir | files | size (bytes) | directory digest (sha256) |
|---|---|---|---|---|
| `260407_183223_056788` | `PriorNetwork_14` | 25 | 53410618 | `8c92bee13863920375083654d6d0a52a707e4ca8777805a19b1a39d556870877` |
| `260414_072827_774853` | `PriorNetwork_24` | 26 | 53432846 | `4aa8684a0a285ff143421e7f88467cb05fbe68327a9bd9b86433542abad35a60` |
| `260827_153154_576196` | `PriorNetwork_24` | 26 | 84870820 | `218fe445d60124d90df3a91ffc6467ac8ccdfd9157cf6dc8705d45bd90577ba6` |
| `default` | `PriorNetwork_27` | 26 | 84900914 | `c8a4d726f3f56ef724f04fb189cea1762f08fcde93f66ecb06c6d18033ebd426` |

## Aliases

Some configs named a checkpoint by an alternate directory. Each
alias below was compared file by file against the id it maps to:

| alias | id | byte-identical |
|---|---|---|
| `combined_jump` | `260407_183223_056788` | yes |
| `combined_clips_gen0` | `260414_072827_774853` | yes |

## Files

### `260407_183223_056788/PriorNetwork_14`

| file | size (bytes) | sha256 |
|---|---|---|
| `_CHECKPOINT_METADATA` | 467 | `664f3838ae7be804f5a121ae721df392fc031def6fafa6aac21531c4ed9b95d0` |
| `config/metadata` | 3432 | `272cdfdcc6826782d88281b386dbed5cb7fa363f9561d7d94bac47ee20882a70` |
| `policy/_METADATA` | 24267 | `28fdfede5d67e43cd827aeab46f007924d922fb9ca8a9dbb7dfb031c01368de2` |
| `policy/_sharding` | 17173 | `729de312a87db2bc9ebab0c8ca69082decb90fe0071c0f49574a9326d668645d` |
| `policy/array_metadatas/process_0` | 10278 | `aa705125a8e44263774ed578193c9f0b4cf9945da90ee3ca5ea1ef2e252a0bee` |
| `policy/d/45bb6fa41f79512fddbfddca75674458` | 10765 | `2249a461352e065e6a7c3374698612ecc521b9c275917bafb7e51f3264b1ee97` |
| `policy/manifest.ocdbt` | 117 | `eee8b0a3eb5c5b694ce3be30257100d769b89a745b44550ac3425a60d1c932f0` |
| `policy/ocdbt.process_0/d/546e4ae5b1719c327e9e46d5333620a7` | 12832768 | `2b471cb15cf5dbaab53190ac0a430e710896736b7785962bc5b9fc8eeff5562f` |
| `policy/ocdbt.process_0/d/6287fdbfd05cda6e2c6266d172d75dd8` | 4571136 | `9827b89f92432c00346033b9e38ab8594d215e0f7972042bb68328a9cda5f592` |
| `policy/ocdbt.process_0/d/7ce921fb5a62c541eb0445b2f9eae79b` | 784 | `fc6199ecaeb20616d2e6055e3d4e9c8b68cd886ec809dfd4f41b78fee8332ed5` |
| `policy/ocdbt.process_0/d/cefd287798cad265935e775b097f9870` | 190 | `cd4f621468e4d6e7312aa02e6208a9a3086b60013abf1953bde8f1ff1b56c669` |
| `policy/ocdbt.process_0/d/d83032b08eeb83aa07b42fc1c2960d76` | 2997 | `77773e0c0efd58e526f4b6090078c6aa6ebf47124c949aa2980acda8a5765340` |
| `policy/ocdbt.process_0/manifest.ocdbt` | 308 | `25251a2a0c66995318c55bf8dadbfbbac4669d70502a6ee46045154e43f379ab` |
| `train_state/_METADATA` | 38522 | `565c64db02ba209ce0a41ef6e22459bcd11919524462fbdecfaeecbdcb2514d4` |
| `train_state/_sharding` | 26593 | `b9b360c387f45245f196a735cb7110a71cc0c0953f596085ac8753a0da5943ec` |
| `train_state/array_metadatas/process_0` | 16168 | `358692479195970b90666f47e45aecc35beade3324ae6ae9a45365a255286851` |
| `train_state/d/acab20f64415968809ffe2f554b3f49f` | 11441 | `a4de9127945f31607e6eb278356d609c1a25512711ba415a26e00ad4ef6ce439` |
| `train_state/manifest.ocdbt` | 117 | `b80171eb6d6d2f5efac5499b7aa4e3380cbd72b5b6feeb1f9ff388929f29e758` |
| `train_state/ocdbt.process_0/d/2910dc1a20fd4be50e27368ba02f3a5b` | 892 | `25b44463ffc9cf6b059abcf129832066eb76caad2a90003b046e7aff0d666e0e` |
| `train_state/ocdbt.process_0/d/4fe00f3b9d4918f98dadff08f5c1819a` | 23904256 | `511814df46481deb045e0b37bddb43f05845919af5dc183f53103f53b2fb39ea` |
| `train_state/ocdbt.process_0/d/5ecf2554c332aae79e731dd4c2d83141` | 4846 | `c4663ed131adaa403d5384344b4d323403539de01749b53b10a305d06acd8be7` |
| `train_state/ocdbt.process_0/d/b29ec8ee62545f8cbe6382ac43d8d05d` | 922 | `4b244c0eb7ba44051fac3d86212cb1bbb0c2b0316c484032e2e47f1be10673ab` |
| `train_state/ocdbt.process_0/d/ce3bb3aff3b55feee2bfddf74f4ae539` | 11931648 | `419878feac856dc47c8d826de69fff56a67e915b1ea5d2f82b324cfdbd1016d0` |
| `train_state/ocdbt.process_0/d/ee6742f542536e79e5ecd3e73ce1683c` | 187 | `967e698ee3d23a9b60e1615feff7e427a4316b90ac053b26adcec8a44b8b5e99` |
| `train_state/ocdbt.process_0/manifest.ocdbt` | 344 | `8381f0998f55371b8998600fccf3d41c575e9b6d92b3b13abeff2d6fe5bceb95` |

### `260414_072827_774853/PriorNetwork_24`

| file | size (bytes) | sha256 |
|---|---|---|
| `_CHECKPOINT_METADATA` | 467 | `612d94935ffbc1aa650e64664b8e7b85e1d5f3d15aeef760a10bc37d752c8525` |
| `config/metadata` | 3442 | `5f23c424c58b3430908d3c60b9b60297afc22089bb8070e625189385c0533510` |
| `policy/_METADATA` | 24267 | `28fdfede5d67e43cd827aeab46f007924d922fb9ca8a9dbb7dfb031c01368de2` |
| `policy/_sharding` | 17173 | `729de312a87db2bc9ebab0c8ca69082decb90fe0071c0f49574a9326d668645d` |
| `policy/array_metadatas/process_0` | 10278 | `aa705125a8e44263774ed578193c9f0b4cf9945da90ee3ca5ea1ef2e252a0bee` |
| `policy/d/ed62703f695a8147db1061ce8e668860` | 10740 | `868e5694536e92ed9f938debe26a33b1e533be851636973649fd8e15296e7ba0` |
| `policy/manifest.ocdbt` | 117 | `80e37fbf87ce7296e9c96ecbbc0495c7cb458d9dd8f3596d33bbbbbf117a1999` |
| `policy/ocdbt.process_0/d/40d17f74785d31878a5e48a1da0f851a` | 15441920 | `27d9d1d5af163db3315db5c44bc323028643895e6f4a3191827515e670aa56d9` |
| `policy/ocdbt.process_0/d/73e25dc45b945e43d67cb694b39f422e` | 784 | `fc6199ecaeb20616d2e6055e3d4e9c8b68cd886ec809dfd4f41b78fee8332ed5` |
| `policy/ocdbt.process_0/d/7b11acab22cfd1a9595c54370ade5f0c` | 1968302 | `410adc38c526aeec3286ad3be1c4c2dabbbd34bb1638d917d965641d43eaca06` |
| `policy/ocdbt.process_0/d/b886844f24a81a07cb6477921068cace` | 189 | `e99793c3e0680f118461fcd782e845a6f1fc0ae3db1ed2ec109b37123bad86e1` |
| `policy/ocdbt.process_0/d/e8996b559486551a9df27df26ddf2748` | 809 | `b8e27940f835f46e00b6412144d412320511c2f339ca95fb7d50664c137e222d` |
| `policy/ocdbt.process_0/manifest.ocdbt` | 307 | `5f1fdcad1d23c0c439ee40558c32014a3db03b46d7891682d9ce981c3c9b0430` |
| `train_state/_METADATA` | 38522 | `565c64db02ba209ce0a41ef6e22459bcd11919524462fbdecfaeecbdcb2514d4` |
| `train_state/_sharding` | 26593 | `b9b360c387f45245f196a735cb7110a71cc0c0953f596085ac8753a0da5943ec` |
| `train_state/array_metadatas/process_0` | 16168 | `358692479195970b90666f47e45aecc35beade3324ae6ae9a45365a255286851` |
| `train_state/d/e37fdd740c281e4710b3213e7ba77ac8` | 11467 | `4e37a7eceefda2a68433f5e1290db0210a5c10a2ea3d1fff180c23dea437e369` |
| `train_state/manifest.ocdbt` | 117 | `51cc1ab1f97767f5ac8e1a634ddc11e0196d3403cb79d97b23d34a549c6db396` |
| `train_state/ocdbt.process_0/d/08b2ba5c97008504846a6d992ccd7e98` | 892 | `25b44463ffc9cf6b059abcf129832066eb76caad2a90003b046e7aff0d666e0e` |
| `train_state/ocdbt.process_0/d/3e219e4e3400df83d57f91729b2b3d76` | 239 | `1b3b38f4c42989cf66283bab9477cc2c1d183f2f94c8fc0b68cf59a01f91606d` |
| `train_state/ocdbt.process_0/d/759354cd7f3dac8020e0e8c8a8b21edf` | 20897792 | `890feea8fc636090dbb92db1c896d6fd94c868cba666fe56de8e93b4830fe8e7` |
| `train_state/ocdbt.process_0/d/968aba2a7072a593826320d1fcb336da` | 5599232 | `a8a4641c5c337dca93dc16c08f5028444227bde35fa903120ab6213c225d1ce1` |
| `train_state/ocdbt.process_0/d/9b2881df82f1a84acc126765f1aa6c1b` | 1944066 | `7f53098359964e763342059fe0c6577b83dedbd0b3a2009b65e84e536ac40c11` |
| `train_state/ocdbt.process_0/d/ec83404b22b64f463be9fefd6a210dca` | 7413760 | `65a42e2c724bb55eda2fb9072e995e23409643d07c9f0e1867d25d05ef7e21c2` |
| `train_state/ocdbt.process_0/d/ecc0ab2ba309a1ea3f1e5a4ce2fa1e83` | 4806 | `d3013c4b4a3e93cee3e93660252657a7b90f70eadac751de993ece8ad568bef4` |
| `train_state/ocdbt.process_0/manifest.ocdbt` | 397 | `0232176d0c1ee74c913d07788b8325db7c7c0ce4cab41d355c786e4a738db6f4` |

### `260827_153154_576196/PriorNetwork_24`

| file | size (bytes) | sha256 |
|---|---|---|
| `_CHECKPOINT_METADATA` | 467 | `6360e1eea908548e7661e83a1afa048e901688622e6e1162bb9b5c0b406b1a5d` |
| `config/metadata` | 3521 | `755a0675970e68e8cf3bf0e5898f78edfdcf53ce10b213373e9a0c49902abcf9` |
| `policy/_METADATA` | 18889 | `c79fc77aa1e3dd09dfb9b17c3f6ede4b2d6dcdf597292c5c51af066103295dbf` |
| `policy/_sharding` | 13429 | `1d5bb673789aebb71d69c2aee95cb8413d00f7093516a9c7058ca40b1261460b` |
| `policy/array_metadatas/process_0` | 8078 | `8e13ced0a47dd6d90b4bf8f607be6d7016c4a054a72ec7212ecd1c1fa454f087` |
| `policy/d/519ad56e70446f67a9830833b68b9bce` | 1996 | `b106c2af312f6f26b4a9cdc83a9af100511650886ab3c7ff7e74c60c2171b61f` |
| `policy/manifest.ocdbt` | 116 | `4d1e4942e9a599e99da2cbf39d6076a1f11483ca791f306e8a40d3ce3977f981` |
| `policy/ocdbt.process_0/d/1132e1680ffd2ae6699b0e8a3272a5f0` | 18022400 | `7080dc11ba6223e5aeeda664e31b15de4b6f82cf5b4ebac0527a14749d7e13c3` |
| `policy/ocdbt.process_0/d/1f782bc749f1415d6c8f5d3ec59bcf2a` | 3288 | `09a20d68698abe8917261292a1f8800e65eceaca4e315c0546d0d8e8c5669509` |
| `policy/ocdbt.process_0/d/a2b5fa8fc0c5e7ed587df07f90955bbd` | 190 | `cd4f621468e4d6e7312aa02e6208a9a3086b60013abf1953bde8f1ff1b56c669` |
| `policy/ocdbt.process_0/d/b81a3e4427dae45717ed0dc8ac010824` | 11632640 | `d83858b53f740909b5701548ea3891b6c3c04cbae1b4c82682cb3efd0473b85a` |
| `policy/ocdbt.process_0/d/ba57315f015eabdc2ed681aa751b0a7c` | 806 | `83cb072b9e5ae95807578505d07f886b0e55c8b0f2aed1dd62175eab70148ceb` |
| `policy/ocdbt.process_0/d/eda5990fcf58436ce683aaf12b7fd13c` | 3523828 | `e6a6815dee6e18f33792c197bd5ba6c0d83fb641a15fcf0560ac3c8cdf6ffc49` |
| `policy/ocdbt.process_0/manifest.ocdbt` | 348 | `29bb1eab8780b390645194f9bcfe620bfabb1dd9a42ffc4bb328f8f2e41f2abd` |
| `train_state/_METADATA` | 33944 | `539e25c7d5dbee9c6993f18ae3ba3d1a4e6d4999129e53a889d284cdabe81e0f` |
| `train_state/_sharding` | 22737 | `a423fa40ef9e9d4885fa54b66e23b13e8987241b3d0630bdc7ed0016bd481b0c` |
| `train_state/array_metadatas/process_0` | 13888 | `58d4cd661fc962ebd836b1ce73a50265476305fd4a6e2709cecadbe67e871c22` |
| `train_state/d/8caad61bb254a43088f04f79ff24f838` | 2602 | `b3012ee29008577bd0ae4e54817bc6e117204e88469de5d3ead9bc3b913e6e9c` |
| `train_state/manifest.ocdbt` | 117 | `1dae81d4e9555ca4c6f6e22098ace33f290aa3f5eb49befca9cc5e8adc64c9e7` |
| `train_state/ocdbt.process_0/d/00442d5a1c730964de72c238bf6ae04c` | 915 | `f2bffeeb875fa7217bf29f5a06b5d1a7f9a7dbb9bd95eeff4aae57a4adb827f6` |
| `train_state/ocdbt.process_0/d/17b83ba011a9be85a04808a212d2d5c4` | 940 | `f7fc32443a736540a0690d1017e6fac89540bd6cc3d8de7c01219cc8b116388b` |
| `train_state/ocdbt.process_0/d/37ae52dae9d3b2e48fa2f715cfd5d91e` | 4833280 | `54d9a6b5f085854d854bda31ba2d10667d79c94d536bd244991ab5a8855a480c` |
| `train_state/ocdbt.process_0/d/81a046fc1dd645ecda5a718dfa00ea2f` | 44785664 | `5e66e3218698c73db02f539c2afe2a128e8956fecceca7a21f05926571b9e275` |
| `train_state/ocdbt.process_0/d/92851cc328efd87758fbb7f97faa15cd` | 1946211 | `8520d252fdd71dafbfa77ab0b7cafb105e12adb57f8f56d4a15b1e0c53566ea4` |
| `train_state/ocdbt.process_0/d/c5bbce77fcb498e7f6d27cde57daae5b` | 187 | `967e698ee3d23a9b60e1615feff7e427a4316b90ac053b26adcec8a44b8b5e99` |
| `train_state/ocdbt.process_0/manifest.ocdbt` | 339 | `8694ff0dfcefab59a54344e5c792620b5e2d28ade2362013a67f78c37795e107` |

### `default/PriorNetwork_27`

| file | size (bytes) | sha256 |
|---|---|---|
| `_CHECKPOINT_METADATA` | 467 | `0dd6a75253cd9db32298780896b7743e4e7437a0e4cb71b3848d24b82679c1b5` |
| `config/metadata` | 3437 | `b8ca0b9aeded5a4eea7b16974471340fd7e84dfd8e68630c52eca8435a35e195` |
| `policy/_METADATA` | 18889 | `c79fc77aa1e3dd09dfb9b17c3f6ede4b2d6dcdf597292c5c51af066103295dbf` |
| `policy/_sharding` | 13429 | `1d5bb673789aebb71d69c2aee95cb8413d00f7093516a9c7058ca40b1261460b` |
| `policy/array_metadatas/process_0` | 8078 | `8e13ced0a47dd6d90b4bf8f607be6d7016c4a054a72ec7212ecd1c1fa454f087` |
| `policy/d/c53de61afea9d3c43e8946fbe661be19` | 1931 | `70a7fd303f76e5cd3588184037dbae8863001a82350819706ec768d73f4cc0f3` |
| `policy/manifest.ocdbt` | 116 | `ca3273277e4016f260cce4257179023d461f03ddf589fdbe9d5408db9f65b398` |
| `policy/ocdbt.process_0/d/1c5a07575f2f51e2ac6f01ab6ed23e0e` | 1966671 | `e59276d69d00eab719bac09c648eb0c41706892ab35b9cf181b8086bf2191230` |
| `policy/ocdbt.process_0/d/5a1e6659aacf2b8767efed64b389b152` | 806 | `83cb072b9e5ae95807578505d07f886b0e55c8b0f2aed1dd62175eab70148ceb` |
| `policy/ocdbt.process_0/d/5c2dcb8cd94d9baf36b12b69bc21c91c` | 31211520 | `bdf97e77f01293e63a386a3747e142587d241efde3a5b85fb2707117ba73ecc3` |
| `policy/ocdbt.process_0/d/8c1c3e96dcf2e3380faece4a75b26da8` | 205 | `6042639ec8ee0c68d47778f4f93dcc0f14c91e66c24f2ac492eb76a1e30f0084` |
| `policy/ocdbt.process_0/d/ae478286d15ca9273c5b52b0c5ba1d3f` | 4886 | `af5369001910c32b8688046ca82794fcfb04ffe97b79a68d0f9ecf801a93681b` |
| `policy/ocdbt.process_0/manifest.ocdbt` | 301 | `4fb25bf9ae7a3b208be64b13f5ce124aaf94a97228697db8e404f4811ce0256b` |
| `train_state/_METADATA` | 33944 | `539e25c7d5dbee9c6993f18ae3ba3d1a4e6d4999129e53a889d284cdabe81e0f` |
| `train_state/_sharding` | 22737 | `a423fa40ef9e9d4885fa54b66e23b13e8987241b3d0630bdc7ed0016bd481b0c` |
| `train_state/array_metadatas/process_0` | 13888 | `58d4cd661fc962ebd836b1ce73a50265476305fd4a6e2709cecadbe67e871c22` |
| `train_state/d/7ef889d7903111c10f89eaec323d475b` | 2646 | `7e877db7df39a6be0c687991f713e65a3d08ca254b98178da5224b35dcbc177c` |
| `train_state/manifest.ocdbt` | 117 | `a796d8c3769f905a4d3e7f8378c45ab0f7304902640c58c480bf199e103b20c3` |
| `train_state/ocdbt.process_0/d/0eef8435c0f12275e0542b75bde5e807` | 713 | `92cc7c1f77f711b17f6514545b9a92cebcb0ac6d3c3adf677d30008582ac5f9c` |
| `train_state/ocdbt.process_0/d/22c634a75b5ae726e5edbda5d57c0266` | 187 | `967e698ee3d23a9b60e1615feff7e427a4316b90ac053b26adcec8a44b8b5e99` |
| `train_state/ocdbt.process_0/d/471162636809f1dac625670cb4c91dfe` | 37072896 | `cde449c26ef3323d70566818510f4bbc9123d0ec6b8c3f29f413775b81d8d74c` |
| `train_state/ocdbt.process_0/d/48cb4d65949b5b40c851a703cba1841d` | 942 | `541aa605efc6084467cd303a491020816cd0e2de9040b22f504703db4cbbc96e` |
| `train_state/ocdbt.process_0/d/507c74c6fa8b7b25b51bee701aeeb854` | 11427840 | `1c86196070fa753e52be06813d1330907298d58d783ad08e7ba95af7027bf885` |
| `train_state/ocdbt.process_0/d/e2072b1c9f268974f6d1928121709333` | 17474 | `84385dc514bf21e666042c7174c838dc1edd79687f2b41c188deb4a304c33b57` |
| `train_state/ocdbt.process_0/d/e6325a3a13ae20d726fa28179ec465e5` | 3076408 | `fb7a2829d9351f0b0502ea9ef3d344938552088fa5a706a2119d530ccf2cce8d` |
| `train_state/ocdbt.process_0/manifest.ocdbt` | 386 | `f24330f56719ba5607f92628b092371a8f6a80c54bc24c58815fb51aa384bae6` |
