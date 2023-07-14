/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
package com.syndrocode.gpt2spanish.di

import com.syndrocode.gpt2spanish.data.autocomplete.AutoCompleteService
import com.syndrocode.gpt2spanish.data.autocomplete.AutoCompleteServiceImpl
import com.mediamonks.wordfilter.LanguageChecker
import com.mediamonks.wordfilter.LanguageCheckerImpl
import org.koin.android.ext.koin.androidContext
import org.koin.core.module.dsl.singleOf
import org.koin.dsl.module

val appModule = module {
    single<AutoCompleteService> {
        AutoCompleteServiceImpl(
            context = androidContext(),
            languageChecker = get()
        )
    }

    singleOf<LanguageChecker>(::LanguageCheckerImpl)
}
